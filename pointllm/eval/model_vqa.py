import os
import math
import json
import torch
import argparse
import shortuuid
from tqdm import tqdm
from transformers import AutoTokenizer
from pointllm.model import *

from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model import *
from pointllm.model.utils import KeywordsStoppingCriteria
from pointllm.data import ObjectPointCloudDataset
import transformers
import numpy as np

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

def init_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    # * print the model_name (get the basename)
    print(f'[INFO] Model name: {os.path.basename(model_name)}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = PointLLMLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False, use_cache=True, torch_dtype=torch.bfloat16).cuda()
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    conv_mode = "vicuna_v1_1"

    conv = conv_templates[conv_mode].copy()

    return model, tokenizer, conv

def pc_norm(pc):
    """ pc: NxC, return NxC """
    xyz = pc[:, :3]
    other_feature = pc[:, 3:]

    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    xyz = xyz / m

    pc = np.concatenate((xyz, other_feature), axis=1)
    return pc

def generate_outputs(model, tokenizer, input_ids, point_clouds, stopping_criteria, do_sample=True, temperature=1.0, top_k=50, max_length=2048, top_p=0.95):
    model.eval() 
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            point_clouds=point_clouds,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            max_length=max_length,
            top_p=top_p,
            stopping_criteria=[stopping_criteria]) # * B, L'

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    outputs = [output.strip() for output in outputs]

    return outputs

def eval_model(args):
    # Model
    disable_torch_init()
    # model_name = get_model_name_from_path(args.model_path)
    model_name = args.model_name.split('/')[-1]
    model, tokenizer, conv = init_model(args)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):

        conv_mode = "vicuna_v1_1"

        conv = conv_templates[conv_mode].copy()

        idx = line["question_id"]
        point_file = line["point"]
        qs = line["text"]
        cur_prompt = qs

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2  

        point_backbone_config = model.get_model().point_backbone_config
        point_token_len = point_backbone_config['point_token_len']
        default_point_patch_token = point_backbone_config['default_point_patch_token']
        default_point_start_token = point_backbone_config['default_point_start_token']
        default_point_end_token = point_backbone_config['default_point_end_token']
        mm_use_point_start_end = point_backbone_config['mm_use_point_start_end']

        if mm_use_point_start_end:
            qs = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + '\n' + qs
        else:
            qs = default_point_patch_token * point_token_len + '\n' + qs

        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        inputs = tokenizer([prompt])
        input_ids_ = torch.as_tensor(inputs.input_ids).cuda() # * tensor of 1, L
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids_)

        point = np.load(os.path.join(args.point_folder, point_file))
        pts_tensor = pc_norm(point)
        pts_tensor = torch.from_numpy(pts_tensor.astype(np.float32))
        pts_tensor = pts_tensor.to(model.device).unsqueeze(0)
        pts_tensor = pts_tensor.to(torch.bfloat16)

        outputs = generate_outputs(model, tokenizer, input_ids_, pts_tensor, stopping_criteria)

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_name", type=str, default="RunsenXu/PointLLM_7B_v1.2") 
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--point-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="tables/answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)