import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import numpy as np
import random
import math

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    #
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

def furthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    B, N, C = xyz.shape
    
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(xyz.device)
    distance = torch.ones(B, N).to(xyz.device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(xyz.device)
    batch_indices = torch.arange(B, dtype=torch.long).to(xyz.device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        dist = dist.to(distance.dtype)
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def fps(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return index_points(xyz, centroids)

class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        # self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        B, N, C = xyz.shape
        if C > 3:
            data = xyz
            xyz = data[:,:,:3]
            rgb = data[:, :, 3:]
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group)  # B G 3

        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center) # B G M
        idx = knn_point(self.group_size, xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        neighborhood_xyz = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood_xyz = neighborhood_xyz.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        if C > 3:
            neighborhood_rgb = rgb.view(batch_size * num_points, -1)[idx, :]
            neighborhood_rgb = neighborhood_rgb.view(batch_size, self.num_group, self.group_size, -1).contiguous()

        # normalize xyz 
        neighborhood_xyz = neighborhood_xyz - center.unsqueeze(2)
        if C > 3:
            neighborhood = torch.cat((neighborhood_xyz, neighborhood_rgb), dim=-1)
        else:
            neighborhood = neighborhood_xyz
        return neighborhood, center

# FPS + k-NN
class FPS_kNN(nn.Module):
    def __init__(self, group_num, k_neighbors):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors = k_neighbors

    def forward(self, xyz, x, rgb):
        # FPS
        fps_idx = furthest_point_sample(xyz.contiguous(), self.group_num).long() 
        lc_xyz = index_points(xyz, fps_idx)
        # 
        lc_x = index_points(x, fps_idx)
        
        lc_rgb = index_points(rgb, fps_idx)
        # kNN
        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx)
        knn_x = index_points(x, knn_idx)
        
        knn_rgb = index_points(rgb, knn_idx)

        return lc_xyz, lc_x, lc_rgb, knn_xyz, knn_x, knn_rgb

# Local Geometry Aggregation
class LGA(nn.Module):
    def __init__(self, out_dim, alpha, beta, vv, LGA_dim):  #dim_expansion, type
        super().__init__()
        alpha, beta = 1, 1
        self.geo_extract = PosE_Geo(3, out_dim, alpha, beta, vv)
        self.param_linear = True
        if LGA_dim == 2:
            self.linear1 = Linear1Layer(out_dim, out_dim, bias=False)
            self.linear2 = []
            self.linear2.append(Linear2Layer(out_dim, bias=True))
            self.linear2 = nn.Sequential(*self.linear2)

    def forward(self, lc_xyz, lc_x, lc_rgb, knn_xyz, knn_x, knn_rgb):
        
        # Normalize x (features) and xyz (coordinates)
        mean_x = lc_x.unsqueeze(dim=-2)
        std_x = torch.std(knn_x - mean_x)
        mean_xyz = lc_xyz.unsqueeze(dim=-2)
        std_xyz = torch.std(knn_xyz - mean_xyz)
        

        knn_x = (knn_x - mean_x) / (std_x + 1e-5)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)

        # Feature Expansion
        B, G, K, C = knn_x.shape
        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)

        # Geometry Extraction
        knn_xyz = knn_xyz.permute(0, 3, 1, 2)
        knn_x = knn_x.permute(0, 3, 1, 2)
        knn_rgb = knn_rgb.permute(0, 3, 1, 2)
        
        if self.param_linear:
            knn_x = self.linear1(knn_x.reshape(B, -1, G*K)).reshape(B, -1, G, K)
        
        knn_x_w = self.geo_extract(knn_xyz, knn_x, knn_rgb)
        
        if self.param_linear:
            for layer in self.linear2:
                knn_x_w = layer(knn_x_w)
        
        return knn_x_w


# Pooling
class Pooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

        self.out_transform = nn.Sequential(
            nn.BatchNorm1d(out_dim),
            nn.GELU())

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(-1)[0] 

        # target_dtype = next(self.out_transform.parameters()).dtype
        # # 将lc_x转换为目标dtype
        # lc_x = lc_x.to(target_dtype)

        lc_x = self.out_transform(lc_x)
        return lc_x


class Linear1Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(Linear1Layer, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


# Linear Layer 2
class Linear2Layer(nn.Module):
    def __init__(self, in_channels, kernel_size=1, groups=1, bias=True):
        super(Linear2Layer, self).__init__()

        self.act = nn.ReLU(inplace=True)
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels/2),
                    kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm2d(int(in_channels/2)),
            self.act
        )
        self.net2 = nn.Sequential(
                nn.Conv2d(in_channels=int(in_channels/2), out_channels=in_channels,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm2d(in_channels)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


# PosE for Local Geometry Extraction
class PosE_Geo(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta, vv):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta
        self.vv = vv

    def forward(self, knn_xyz, knn_x, knn_rgb):
        B, _, G, K = knn_xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)
        

        feat_range = torch.arange(feat_dim).float().cuda()     
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * knn_xyz.unsqueeze(-1), dim_embed)


        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=5).flatten(4)
        position_embed = position_embed.permute(0, 1, 4, 2, 3).reshape(B, self.out_dim, G, K)
        # Weigh
        knn_x_w = knn_x + position_embed
        
        return knn_x_w
        #return knn_x_w.to(knn_x.dtype)


# Non-Parametric Encoder
class EncNP(nn.Module):  
    def __init__(self, input_points, num_stages, embed_dim, k_neighbors, alpha, beta, vv, LGA_dim):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.alpha, self.beta = alpha, beta

        # Raw-point Embedding
        self.raw_point_embed = Linear1Layer(6, self.embed_dim, bias=False)

        self.FPS_kNN_list = nn.ModuleList() # FPS, kNN
        self.LGA_list = nn.ModuleList() # Local Geometry Aggregation
        self.Pooling_list = nn.ModuleList() # Pooling
        
        out_dim = self.embed_dim
        group_num = self.input_points

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            if LGA_dim[i] == 2 or LGA_dim[i] == 1:
                out_dim = out_dim * 2
                group_num = group_num // 2
            self.FPS_kNN_list.append(FPS_kNN(group_num, k_neighbors))
            self.LGA_list.append(LGA(out_dim, self.alpha, self.beta, vv, LGA_dim[i]))
            self.Pooling_list.append(Pooling(out_dim))
        


    def forward(self, xyz, x, rgb, rgbx, xyz_ori, x_ori):

        # Raw-point Embedding
        x = self.raw_point_embed(xyz_ori)
        
        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            # FPS, kNN
            xyz, lc_x, rgb, knn_xyz, knn_x, knn_rgb = self.FPS_kNN_list[i](xyz, x.permute(0, 2, 1), rgb)
            
            # Local Geometry Aggregation
            knn_x_w = self.LGA_list[i](xyz, lc_x, rgb, knn_xyz, knn_x, knn_rgb)
            
            # Pooling
            x = self.Pooling_list[i](knn_x_w)
            

        return x, knn_xyz, xyz


# Non-Parametric Network
class PointNN(nn.Module):
    def __init__(self, config, use_max_pool=True):
        super().__init__()

        self.use_max_pool = use_max_pool
        self.config = config
        # self.input_points = 1024
        self.input_points = config.input_points
        self.num_stages = config.num_stages
        self.embed_dim = config.embed_dim
        self.pos_init_dim = config.pos_init_dim
        self.k_neighbors = config.group_size
        self.beta = 1000
        self.alpha = 100
        self.LGA_dim = config.LGA_dim
        self.point_dims = config.point_dims
        self.vv = torch.randn(1, 5000)

        if self.LGA_dim[-1] != 3:
            self.out_dim = self.embed_dim*(2**self.num_stages)
        else:
            self.out_dim = self.embed_dim*(2**(self.num_stages-1))
            
        self.EncNP = EncNP(self.input_points, self.num_stages, self.embed_dim, self.k_neighbors, self.alpha, self.beta, self.vv, self.LGA_dim)

        self.class_embedding = nn.Parameter(torch.randn(1,1,self.out_dim))


    def forward(self, x, pos_head_type=None):
        # xyz: point coordinates
        # x: point features

        pos, rgb = x[..., :3], x[..., 3:]
        xyz, pos_x = pos, pos.permute(0, 2, 1)
        rgb, rgbx = rgb, rgb.permute(0, 2, 1)
        xyz_ori = x.permute(0, 2, 1)
        
        x, knn_xyz, xyz = self.EncNP(xyz, pos_x, rgb, rgbx, xyz_ori, x) #mae
        x = x.transpose(1, 2).contiguous()
        # Non-Parametric Encoder
        class_embed = self.class_embedding.expand(x.size(0), -1, -1).to(dtype=x.dtype)

        feature_final = torch.cat([class_embed, x], dim=1)

        return feature_final, knn_xyz, xyz, None #mae

    @property
    def dtype(self):
        return self.EncNP.raw_point_embed.net[0].weight.dtype

    @property
    def device(self):
        return self.EncNP.raw_point_embed.net[0].weight.device