#    Copyright 2023 Runsen Xu

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from .utils import *
from pointllm.utils import *

from contextlib import nullcontext
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

import os
from pointllm.model.loss import *
import numpy as np

# * add logger
import logging
logger = logging.getLogger(__name__)

class PointLLMConfig(LlamaConfig):
    model_type = "pointllm"

class PointLLMLlamaModel(LlamaModel):
    config_class = PointLLMConfig 

    def __init__(self, config: LlamaConfig):
        super(PointLLMLlamaModel, self).__init__(config)

        self.point_backbone_type = config.point_backbone
        logger.info(f"Using {self.point_backbone_type}.")
        
        if self.point_backbone_type == "PointNN":
            from pointllm.model import PointNN
            # address of config file, in the same dir of this file
            point_bert_config_name = getattr(config, "point_backbone_config_name", "PointTransformer_8192point_2layer") # * default for v1.2, v1.1 uses PointTransformer_base_8192point.yaml
            point_bert_config_addr = os.path.join(os.path.dirname(__file__), "pointnn", f"{point_bert_config_name}.yaml")
            print(f"Loading PointNN config from {point_bert_config_addr}.")
            point_bert_config = cfg_from_yaml_file(point_bert_config_addr)
            if getattr(config, "use_color", False):
                point_bert_config.model.point_dims = 6
            use_max_pool = getattr(point_bert_config.model, "use_max_pool", False) # * default is false
            
            point_bert_config.model.group_size = config.point_pn_params['group_size']
            point_bert_config.model.num_stages = config.point_pn_params['num_stages']
            point_bert_config.model.embed_dim = config.point_pn_params['embed_dim']
            point_bert_config.model.LGA_dim = config.point_pn_params['LGA_dim']
            point_bert_config.model.input_points = config.point_pn_params['input_points']

            self.point_backbone = PointNN(point_bert_config.model, use_max_pool=use_max_pool)
            logger.info(f"Using {self.point_backbone.point_dims} dim of points.") 


            if point_bert_config.model.LGA_dim[-1] != 3:
                self.point_backbone_config = {
                    "point_cloud_dim": point_bert_config.model.point_dims,
                    "backbone_output_dim": point_bert_config.model.embed_dim*(2**point_bert_config.model.num_stages) if not use_max_pool else point_bert_config.model.encoder_dims * 2,
                    "project_output_dim": self.config.hidden_size,
                    "point_token_len": (point_bert_config.model.input_points//(2**point_bert_config.model.num_stages)) + 1 if not use_max_pool else 1, # * number of output features, with cls token
                    "mm_use_point_start_end": self.config.mm_use_point_start_end,
                    "projection_hidden_layer": point_bert_config.model.get('projection_hidden_layer', 0),
                    "use_max_pool": use_max_pool
                }
            else:
                self.point_backbone_config = {
                    "point_cloud_dim": point_bert_config.model.point_dims,
                    "backbone_output_dim": point_bert_config.model.embed_dim*(2**(point_bert_config.model.num_stages-1)) if not use_max_pool else point_bert_config.model.encoder_dims * 2,
                    "project_output_dim": self.config.hidden_size,
                    "point_token_len": (point_bert_config.model.input_points//(2**point_bert_config.model.num_stages)) + 1 if not use_max_pool else 1, # * number of output features, with cls token
                    "mm_use_point_start_end": self.config.mm_use_point_start_end,
                    "projection_hidden_layer": point_bert_config.model.get('projection_hidden_layer', 0),
                    "use_max_pool": use_max_pool
                }
            if point_bert_config.model.get('projection_hidden_layer', 0) > 0:
                self.point_backbone_config["projection_hidden_dim"] = point_bert_config.model.projection_hidden_dim # a list
            
            logger.info(f"Use max pool is {use_max_pool}. Number of point token is {self.point_backbone_config['point_token_len']}.")

        # * print relevant info with projection layers
        backbone_output_dim = self.point_backbone_config["backbone_output_dim"]
        logger.info(f"Point backbone output dim: {backbone_output_dim}.")
        logger.info(f"Use {self.point_backbone_config['projection_hidden_layer']} projection hiddent layers.")
        if self.point_backbone_config['projection_hidden_layer'] > 0:
            # Add projection layer with linear layers and GELU activation
            projection_layers = []
            last_dim = backbone_output_dim
            for i in range(point_bert_config.model.projection_hidden_layer):
                projection_layers.append(nn.Linear(last_dim, self.point_backbone_config["projection_hidden_dim"][i]))
                projection_layers.append(nn.GELU())
                last_dim = self.point_backbone_config["projection_hidden_dim"][i]

            projection_layers.append(nn.Linear(last_dim, self.point_backbone_config["project_output_dim"]))
            self.point_proj = nn.Sequential(*projection_layers)
            logger.info(f"Each layer with {point_bert_config.model.projection_hidden_dim} hidden units.")
        else:
            # Single layer
            self.point_proj = nn.Linear(backbone_output_dim, self.point_backbone_config['project_output_dim'])
        logger.info(f"Point projector output dim: {self.point_backbone_config['project_output_dim']}.")

        self.fix_pointnet = False
        self.fix_llm = False
        if config.point_pn_params['mae_fp'] != 2:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, config.point_pn_params['mask_dim']))   # mae
        self.mask_ratio = config.point_pn_params['mask_ratio']
        self.config = config
        self.recon_fp = config.point_pn_params['recon_fp']
        self.mae_fp = config.point_pn_params['mae_fp']
        self.recon_pos = config.point_pn_params['recon_pos']
        self.mask_dim = config.point_pn_params['mask_dim']

        self.pos_embed_mae = False
        self.pos_embed_type = config.point_pn_params['pos_embed_mae']
        if config.point_pn_params['pos_embed_mae'] != 0:
            self.pos_embed_mae = True
            
            self.decoder_pos_embed = nn.Sequential(
                nn.Linear(3, 128),
                nn.GELU(),
                nn.Linear(128, config.point_pn_params['pos_embed_dim'])
            )

    def load_point_backbone_checkpoint(self, checkpoint_path=None):
        self.point_backbone.load_checkpoint(self.config.point_backbone_ckpt if checkpoint_path is None else checkpoint_path)
    
    # mae
    def _mask_center_rand(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        G = G - 1  
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)
        cls_token = torch.from_numpy(np.zeros([B, 1])).to(torch.bool)
        end_mask = torch.cat([cls_token,overall_mask],dim=1)
        return end_mask.to(center.device) # B G
    # contrast 
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        point_clouds: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        point_backbone = getattr(self, 'point_backbone', None)
        point_backbone_config = getattr(self, 'point_backbone_config', None)
        point_token_start = []
        xyz = []
        if point_backbone is not None and (input_ids.shape[1] != 1 or self.training) and point_clouds is not None:
            # print('success')
            with torch.no_grad() if self.fix_pointnet else nullcontext():
                if self.fix_pointnet:
                    self.point_backbone.eval()
                if type(point_clouds) is list:
                    # * variable numbers of points
                    point_features_ori = [] 
                    for point_cloud in point_clouds: # * iterate over batch
                        point_feature, knn_xyz, xyz, position_embed = self.point_backbone(point_cloud.unsqueeze(0))[0]   # mae
                        point_features_ori.append(point_feature)
                else:
                    point_features_ori, knn_xyz, xyz, position_embed = self.point_backbone(point_clouds)
            
            mae_gts = None  #patch.feature
            recon_gts_points = None   # patch
            recon_gts_features = None  # feature
            bool_masked_pos = None
            point_features_vis = 0
            point_features_mask = 0
            pos_emd_vis = None
            pos_emd_mask = None
            if self.mask_dim == self.point_backbone_config['backbone_output_dim'] and (self.config.point_pn_params['mask_ratio'] == 0.3 or self.config.point_pn_params['mask_ratio'] == 0.6):
                bool_masked_pos = self._mask_center_rand(point_features_ori, noaug = False) #  bool_masked_pos B G
                batch_size, seq_len, _ = point_features_ori.size() # 2304
                point_features_vis = point_features_ori[~bool_masked_pos].reshape(batch_size, -1, self.config.point_pn_params['mask_dim'])
                recon_gts_points = knn_xyz[~bool_masked_pos[:,1:]].reshape(batch_size, -1, knn_xyz.size()[2], knn_xyz.size()[3])
                if self.pos_embed_mae:               
                    pos_emd_vis = self.decoder_pos_embed(xyz[~bool_masked_pos[:,1:]]).reshape(batch_size, -1, self.config.point_pn_params['pos_embed_dim'])
                    pos_emd_mask = self.decoder_pos_embed(xyz[bool_masked_pos[:,1:]]).reshape(batch_size, -1, self.config.point_pn_params['pos_embed_dim'])

                if self.mae_fp==0: # patch
                    mae_gts = knn_xyz[bool_masked_pos[:,1:]].reshape(batch_size, -1, knn_xyz.size()[2], knn_xyz.size()[3])
                elif self.mae_fp==1: # feature
                    mae_gts = point_features_ori[bool_masked_pos].reshape(batch_size, -1, self.config.point_pn_params['mask_dim'])

                if isinstance(mae_gts, list):
                    mask_token = self.mask_token.expand(batch_size, mae_gts[0].size()[1], -1)
                else:
                    mask_token = self.mask_token.expand(batch_size, mae_gts.size()[1], -1)
                # 
                if self.training:
                    point_features_ori = torch.cat([point_features_vis,mask_token],dim=1)
                    if self.recon_pos == 0:
                        recon_gts_features = point_features_vis
                else:
                    point_features_ori = point_features_ori


            if type(point_clouds) is list: #mae
                point_features_ori = [self.point_proj(point_feature) for point_feature in point_features_ori]
            else:
                point_features_ori = self.point_proj(point_features_ori) # 4096

            if self.config.point_pn_params['mask_dim'] == self.point_backbone_config["project_output_dim"] and (self.config.point_pn_params['mask_ratio'] == 0.6 or self.config.point_pn_params['mask_ratio'] == 0.3):
                # mae
                # point_features = point_features_ori
                bool_masked_pos = self._mask_center_rand(point_features_ori, noaug = False) #  bool_masked_pos B G
                batch_size, seq_len, _ = point_features_ori.size()
                point_features_vis = point_features_ori[~bool_masked_pos].reshape(batch_size, -1, self.config.point_pn_params['mask_dim'])

                recon_gts_points = knn_xyz[~bool_masked_pos[:,1:]].reshape(batch_size, -1, knn_xyz.size()[2], knn_xyz.size()[3])

                if self.pos_embed_mae:  
                    pos_emd_vis = self.decoder_pos_embed(xyz[~bool_masked_pos[:,1:]]).reshape(batch_size, -1, self.config.point_pn_params['pos_embed_dim'])
                    pos_emd_mask = self.decoder_pos_embed(xyz[bool_masked_pos[:,1:]]).reshape(batch_size, -1, self.config.point_pn_params['pos_embed_dim'])

                if self.mae_fp==0:
                    mae_gts = knn_xyz[bool_masked_pos[:,1:]].reshape(batch_size, -1, knn_xyz.size()[2], knn_xyz.size()[3])
                elif self.mae_fp==1:
                    mae_gts = point_features_ori[bool_masked_pos].reshape(batch_size, -1, self.config.point_pn_params['mask_dim'])

                if isinstance(mae_gts, list):
                    mask_token = self.mask_token.expand(batch_size, mae_gts[0].size()[1], -1)
                else:
                    mask_token = self.mask_token.expand(batch_size, mae_gts.size()[1], -1)
                # 
                if self.training:
                    point_features = torch.cat([point_features_vis,mask_token],dim=1)
                else:
                    point_features = point_features_ori

            if self.recon_pos == 1:  
                if bool_masked_pos is not None:
                    batch_size, seq_len, _ = point_features_ori.size() # 2304
                    recon_gts_features = point_features_ori[~bool_masked_pos].reshape(batch_size, -1, 4096)
                else:
                    recon_gts_features = point_features_ori

            if (self.mask_dim == self.point_backbone_config['backbone_output_dim']) or (self.config.point_pn_params['mask_ratio'] != 0.6 and self.config.point_pn_params['mask_ratio'] != 0.3):
                point_features = point_features_ori

            dummy_point_features = torch.zeros(point_backbone_config['point_token_len'], point_backbone_config['backbone_output_dim'], device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_point_features = self.point_proj(dummy_point_features)

            new_input_embeds = []  
            cur_point_idx = 0
            mask_start = [] 
            full_reconstruction_start = []
            vis_pos_start = []
            mask_pos_start = []

            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds): # * input_ids: B, L; input_embeds: B, L, C
                if (cur_input_ids == point_backbone_config['point_patch_token']).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_point_features).sum() # * do nothing
                    new_input_embeds.append(cur_input_embeds)
                    cur_point_idx += 1
                    continue
                cur_point_features = point_features[cur_point_idx].to(device=cur_input_embeds.device)
                num_patches = cur_point_features.shape[0] # * number of point tokens
                if point_backbone_config['mm_use_point_start_end']:
                    if (cur_input_ids == point_backbone_config["point_start_token"]).sum() != (cur_input_ids == point_backbone_config["point_end_token"]).sum():
                        raise ValueError("The number of point start tokens and point end tokens should be the same.")
                    point_start_tokens = torch.where(cur_input_ids == point_backbone_config["point_start_token"])[0]
                    point_token_start.append(point_start_tokens+1+1)
    
                    if isinstance(point_features_vis, int):
                        mask_start.append(point_start_tokens+1)  # mae  
                        vis_pos_start.append(point_start_tokens+2)  
                        mask_pos_start.append(point_start_tokens+2)
                    else:
                        mask_start.append(point_start_tokens + point_features_vis.size()[1]+1)
                        vis_pos_start.append(point_start_tokens+2)
                        mask_pos_start.append(point_start_tokens+1+point_features_vis.size()[1])

                    
                    if self.recon_fp == 1:
                        full_reconstruction_start.append(point_start_tokens + 1) # t->t  full reconstruction
                    else:
                        full_reconstruction_start.append(point_start_tokens + 1 + 1) # t->t  full reconstruction
                   
                    for point_start_token_pos in point_start_tokens:
                        if cur_input_ids[point_start_token_pos + num_patches + 1] != point_backbone_config["point_end_token"]:
                            raise ValueError("The point end token should follow the point start token.")
                        if orig_embeds_params is not None: # * will not update the original embeddings except for POINT_START_TOKEN and POINT_END_TOKEN
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:point_start_token_pos].detach(), cur_input_embeds[point_start_token_pos:point_start_token_pos+1], cur_point_features, cur_input_embeds[point_start_token_pos + num_patches + 1:point_start_token_pos + num_patches + 2], cur_input_embeds[point_start_token_pos + num_patches + 2:].detach()), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:point_start_token_pos+1], cur_point_features, cur_input_embeds[point_start_token_pos + num_patches + 1:]), dim=0)
                        cur_point_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    if (cur_input_ids == point_backbone_config["point_patch_token"]).sum() != num_patches:
                        raise ValueError("The number of point patch tokens should be the same as the number of point patches.")
                    masked_indices = torch.where(cur_input_ids == point_backbone_config["point_patch_token"])[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The point patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_point_features, cur_input_embeds[mask_index_start+num_patches:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_point_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    cur_point_idx += 1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        if self.training:
            is_vis_pos_start_uniform = all(v == vis_pos_start[0] for v in vis_pos_start)
            is_mask_pos_start_uniform = all(m == mask_pos_start[0] for m in mask_pos_start)

            vis_pos_start_value = vis_pos_start[0] if is_vis_pos_start_uniform else None
            mask_pos_start_value = mask_pos_start[0] if is_mask_pos_start_uniform else None
        else:
            vis_pos_start_value = None
            mask_pos_start_value = None 


        if self.pos_embed_mae:
            output = super(PointLLMLlamaModel, self).forward(
                input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, use_cache=use_cache,
                output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
        else:
            output = super(PointLLMLlamaModel, self).forward(
                input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, use_cache=use_cache,
                output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

        if self.training:   
            return output,mae_gts,recon_gts_points,recon_gts_features,mask_start,full_reconstruction_star
        else:
            return output

class PointLLMLlamaForCausalLM(LlamaForCausalLM):
    config_class = PointLLMConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = PointLLMLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config
        self.mask_ratio = config.point_pn_params['mask_ratio']
        self.mask_dim = config.point_pn_params['mask_dim']
        self.config = config
        self.recon_fp = config.point_pn_params['recon_fp']
        self.mae_fp = config.point_pn_params['mae_fp']
        self.recon_pos = config.point_pn_params['recon_pos']
        self.cd_loss = ChamferDistanceL2().cuda()

        # mae 
        if self.mae_fp ==0:
            self.mae_predict_head =  nn.Sequential(
                nn.Linear(4096,2048),
                nn.GELU(),
                nn.Linear(2048, 1024),
                nn.GELU(),
                nn.Linear(1024, config.point_pn_params['group_size']*3)
            )
        elif self.mae_fp == 1:
            if self.mask_dim==4096:
                self.mae_predict_head =  nn.Sequential(
                    nn.Linear(4096,1024),
                    nn.GELU(),
                    nn.Linear(1024, 4096)
                )
            else:
                self.mae_predict_head =  nn.Sequential(
                    nn.Linear(4096,self.mask_dim),
                    nn.GELU(),
                    nn.Linear(self.mask_dim, self.mask_dim)
                )

        # recon
        if self.recon_fp==0:
            self.recon_predict_head =  nn.Sequential(
                nn.Linear(4096,2048),
                nn.GELU(),
                nn.Linear(2048, 1024),
                nn.GELU(),
                nn.Linear(1024, config.point_pn_params['group_size']*3)
            )
        elif self.recon_fp==1:
            if self.recon_pos == 0:
                self.recon_predict_head =  nn.Sequential(
                    nn.Linear(4096,2304),
                    nn.GELU(),
                    nn.Linear(2304, 2304)
                )
            elif self.recon_pos == 1:
                self.recon_predict_head =  nn.Sequential(
                    nn.Linear(4096,1024),
                    nn.GELU(),
                    nn.Linear(1024, 4096)
                )

        # Initialize weights and apply final processing
        self.post_init()

    def cos_loss(self, pred_feature, clip_feature):
        loss_func = nn.CosineSimilarity(dim=-1)
        return 1 - loss_func(pred_feature, clip_feature).mean()
    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None, # * control whether to return past_key_values
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        point_clouds: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.training:
            outputs,mae_gts,recon_gts_points,recon_gts_features,mask_start,full_reconstruction_start  = self.model(  # mae 
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                point_clouds=point_clouds
            )
        else:
            outputs = self.model(  # mae 
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                point_clouds=point_clouds
            )

        hidden_states = outputs[0]

        if self.training and (self.mask_ratio==0.3 or self.mask_ratio==0.6):    
            if isinstance(mae_gts, list):
                gts_point_size = mae_gts[0].size()[1]
            else:
                gts_point_size = mae_gts.size()[1]           # mae 
            mask_tokens_list = []
            for i in range(hidden_states.size()[0]):
                mask_tokens_list.append(hidden_states[i,mask_start[i]:mask_start[i]+gts_point_size,:])
            mask_tokens = torch.stack(mask_tokens_list,dim=0)
            if self.mae_fp==0:
                B,G,_,_ = mae_gts.shape
                pred_points = self.mae_predict_head(mask_tokens).reshape(B*G,-1,3).float()
                mae_gts = mae_gts.reshape(B*G,-1,3).float()
                mae_loss = self.cd_loss(pred_points,mae_gts)
            elif self.mae_fp==1:
                B,G,_ = mae_gts.shape
                pred_points = self.mae_predict_head(mask_tokens).reshape(B, G, self.config.point_pn_params['mask_dim']).float()
                mae_gts = mae_gts.reshape(B, G, self.config.point_pn_params['mask_dim']).float()
                if self.config.point_pn_params['mae_feature']==0:
                    mae_loss = (pred_points - mae_gts) ** 2
                    mae_loss = mae_loss.mean(dim=-1)  
                    mae_loss = (mae_loss.sum()) / (B*G)  
        else:
            mae_loss = None

        
        if self.training and (self.mask_ratio==0.3 or self.mask_ratio==0.6):    
            reconstruction_tokens_list = []
            for i in range(hidden_states.size()[0]):
                if self.recon_fp==0:
                    reconstruction_tokens_list.append(hidden_states[i,full_reconstruction_start[i]:full_reconstruction_start[i]+recon_gts_points.size()[1],:])
                else:
                    reconstruction_tokens_list.append(hidden_states[i,full_reconstruction_start[i]:full_reconstruction_start[i]+recon_gts_features.size()[1],:])
            reconstruction_tokens = torch.stack(reconstruction_tokens_list,dim=0)

            if self.recon_fp==0:
                B,G,_,_ = recon_gts_points.shape # patch
                pred_points = self.recon_predict_head(reconstruction_tokens).reshape(B*G,-1,3).float()
                recon_gts_points = recon_gts_points.reshape(B*G,-1,3).float()
                full_reconstruction_loss = self.cd_loss(pred_points,recon_gts_points)
            elif self.recon_fp==1:
                B,G,_ = recon_gts_features.shape # center
                pred_points = self.recon_predict_head(reconstruction_tokens).reshape(B, G, -1).float()
                recon_gts_features = recon_gts_features.reshape(B, G, -1).float()
                if self.config.point_pn_params['recon_feature']==0:
                    full_reconstruction_loss = (pred_points - recon_gts_features) ** 2
                    full_reconstruction_loss = full_reconstruction_loss.mean(dim=-1)  
                    full_reconstruction_loss = (full_reconstruction_loss.sum()) / (B*G)  
        else:
            full_reconstruction_loss = None
        
        logits = self.lm_head(hidden_states)

        loss = None

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous() # * B, L, V(32003)
            shift_labels = labels[..., 1:].contiguous() # * B, L
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss_ce = loss_fct(shift_logits, shift_labels)  
            if mae_loss is not None and full_reconstruction_loss is not None:
                loss = loss_ce + mae_loss + full_reconstruction_loss  
            else:
                loss = loss_ce

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "point_clouds": kwargs.get("point_clouds", None),
            }
        )
        return model_inputs

    def initialize_tokenizer_point_backbone_config_wo_embedding(self, tokenizer):
        # * called when stage2 or inference or inference without pre-training, assume tokenizer has point tokens
        config = self.config
        point_backbone_config = self.get_model().point_backbone_config
        mm_use_point_start_end = point_backbone_config['mm_use_point_start_end'] = config.mm_use_point_start_end

        default_point_patch_token = config.DEFAULT_POINT_PATCH_TOKEN

        tokenizer.add_tokens([default_point_patch_token], special_tokens=True)

        # * assert tokenizer has the default_point_patch_token
        point_backbone_config['default_point_patch_token'] = default_point_patch_token
        point_backbone_config['point_patch_token'] = tokenizer.convert_tokens_to_ids([default_point_patch_token])[0]

        if mm_use_point_start_end:
            default_point_start_token = config.DEFAULT_POINT_START_TOKEN
            default_point_end_token = config.DEFAULT_POINT_END_TOKEN
            tokenizer.add_tokens([default_point_start_token, default_point_end_token], special_tokens=True)

            point_backbone_config['default_point_start_token'] = default_point_start_token
            point_backbone_config['default_point_end_token'] = default_point_end_token

            point_backbone_config["point_start_token"] = tokenizer.convert_tokens_to_ids([default_point_start_token])[0]
            point_backbone_config["point_end_token"] = tokenizer.convert_tokens_to_ids([default_point_end_token])[0]
    
    def initialize_tokenizer_point_backbone_config(self, tokenizer, device, fix_llm=True):

        config = self.config
        point_backbone_config = self.get_model().point_backbone_config
        mm_use_point_start_end = point_backbone_config['mm_use_point_start_end'] = config.mm_use_point_start_end

        default_point_patch_token = config.DEFAULT_POINT_PATCH_TOKEN
        point_backbone_config['default_point_patch_token'] = default_point_patch_token
        tokenizer.add_tokens([default_point_patch_token], special_tokens=True) # * no need to update embed since it will be replaced
        self.resize_token_embeddings(len(tokenizer)) # ! resize_token_embeddings will make the tokens trainable again
        point_backbone_config['point_patch_token'] = tokenizer.convert_tokens_to_ids([default_point_patch_token])[0]

        if mm_use_point_start_end:
            default_point_start_token = config.DEFAULT_POINT_START_TOKEN
            default_point_end_token = config.DEFAULT_POINT_END_TOKEN
            point_backbone_config['default_point_start_token'] = default_point_start_token
            point_backbone_config['default_point_end_token'] = default_point_end_token

            num_new_tokens = tokenizer.add_tokens([default_point_start_token, default_point_end_token], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            point_backbone_config["point_start_token"] = tokenizer.convert_tokens_to_ids([default_point_start_token])[0]
            point_backbone_config["point_end_token"] = tokenizer.convert_tokens_to_ids([default_point_end_token])[0]

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

                # need to update the input embeding, but no need to update the output embedding
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                if fix_llm:
                    self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)] # * only tuning the new embeddings
                    for p in self.get_output_embeddings().parameters(): # * the llm head
                        p.requires_grad = False
                    print(f"Setting output embeddings fixed and {num_new_tokens} new tokens' input embeddings trainable.")
                else:
                    self.get_model().orig_embeds_params = None
                    for p in self.get_output_embeddings().parameters():
                        p.requires_grad = True
                    print("Setting output embeddings and all input embeddings trainable.")

AutoConfig.register("pointllm", PointLLMConfig)
AutoModelForCausalLM.register(PointLLMConfig, PointLLMLlamaForCausalLM)
