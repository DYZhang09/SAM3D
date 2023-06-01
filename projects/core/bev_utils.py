from typing import List
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmdet3d.models.layers.fusion_layers.point_fusion import point_sample
from mmdet3d.structures.bbox_3d.utils import get_lidar2img


@MODELS.register_module()
class BEVMapper(BaseModule):
    SUPPORT_MODE = ['binary', 'intensity', 'image_rgb', 'intensity_rgb']

    def __init__(self, 
        point_cloud_range,
        grid_xy_size,
        mode='binary',
        fill_value=255,
        init_cfg = None,
        channel_first=True,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.channel_first = channel_first
        self.point_cloud_range = point_cloud_range
        self.grid_lower_bound = point_cloud_range[:2]  # 2, 
        self.grid_upper_bound = point_cloud_range[3:5]  # 2,
        self.grid_size = grid_xy_size  # 2,
        self.grid_shape = self.calc_grid_shape()  # 2, 

        self.mode = mode
        if self.mode not in BEVMapper.SUPPORT_MODE:
            raise NotImplementedError(f'Not supported mode: {mode}') 
        if self.mode == 'binary':
            self.fill_value = fill_value
        if self.mode == 'intensity_rgb':
            self.remove_ground_points = kwargs.get('remove_ground_points', False)

        self.kwargs = kwargs

    @torch.no_grad()
    def forward(self, batch_input_dict: dict, batch_input_meta: List[dict] = None):
        '''calculate bev maps given inputs

        Args:
            batch_input_dict (dict): the input data, contains keys:
                'points' (List[torch.Tensor]): each entry points of frames of shape N x (3 + C)
                'imgs' (List(torch.Tensor)): each entry contains images of frames of shape  5 x 3 x 1280 x 1920
            batch_input_meta (List[dict]): the meta info of inputs, each entry contains important keys:
                'lidar2img' (np.ndarray): the lidar-to-img projection matrix of shape 4 x 4
                'num_views' (int): the number of image views
                'lidar2cam' (List[List[List]]): len=5, each entry contains a projection matrix (4 x 4 List of List) from lidar to corresponding
                            camera view
                'ego2global' (np.ndarray): the ego-to-global projection matrix of shape 4 x 4
                'ori_cam2img' (List[List[List]]): len=5, each entry contains a projection matrix (4 x 4) from camera to image of each camera view
                'cam2img' (List[List[List]]): the same as 'ori_cam2img'
        '''
        # import ipdb
        # ipdb.set_trace()
        point_clouds = batch_input_dict.get('points')
        point_clouds = torch.stack(point_clouds)
        batch_imgs_list = batch_input_dict.get('img', None)

        bev_map = getattr(self, f'get_{self.mode}_bev')(
            point_clouds = point_clouds,
            batch_imgs_list = batch_imgs_list,
            batch_input_meta = batch_input_meta,
        )  # B x H x W x 3

        if self.channel_first:
            bev_map = bev_map.permute([0, 3, 1, 2]).contiguous()  # B x 3 x H x W 
        return bev_map

    def calc_grid_shape(self):
        shape_x = (self.grid_upper_bound[0] - self.grid_lower_bound[0]) // self.grid_size[0]       
        shape_y = (self.grid_upper_bound[1] - self.grid_lower_bound[1]) // self.grid_size[1]       
        return [int(shape_x) + 1, int(shape_y) + 1]  # +1 to prevent out of bounds

    def ranks_from_coors(self, pc_xy: torch.Tensor):
        '''calculate the rank of grid to which each coordinate belongs

        Args:
            pc_xy (torch.Tensor): point cloud coordinates of shape B x N x 2

        Returns:
            rank_x (torch.LongTensor): the rank of x dimension of shape B x N
            rank_y (torch.LongTensor): the rank of y dimension of shape B x N
        '''
        rank_x = (self.grid_upper_bound[0] - pc_xy[:, :, 0]) / self.grid_size[0]
        rank_y = (self.grid_upper_bound[1] - pc_xy[:, :, 1]) / self.grid_size[1]
        rank_x, rank_y = rank_x.long(), rank_y.long()
        return rank_x, rank_y

    def coors_from_ranks(self, rank_x: torch.LongTensor, rank_y: torch.LongTensor):
        '''calculate the point coordinates from ranks of grids.
            roughly the inverse of ranks_from_coors(), 'roughly' is because of the quantization error

        Args:
            rank_x (torch.LongTensor): the rank of x dimension of shape B x N
            rank_y (torch.LongTensor): the rank of y dimension of shape B x N

        Returns:
            pc_xy (torch.Tensor): point cloud coordinates of shape B x N x 2
        '''
        pc_x = self.grid_upper_bound[0] - (rank_x + 0.5) * self.grid_size[0]
        pc_y = self.grid_upper_bound[1] - (rank_y + 0.5) * self.grid_size[1]
        if isinstance(rank_x, torch.Tensor):
            pc_xy = torch.stack([pc_x, pc_y], dim=-1)  # B x N x 2
        elif isinstance(rank_x, np.ndarray):
            pc_xy = np.stack([pc_x, pc_y], axis=-1)  
        else:
            return pc_x, pc_y
        assert len(pc_xy.shape) == 3
        return pc_xy

    def distinguish_ground(self, point_clouds: torch.Tensor):
        ground_mask = list()
        if len(point_clouds.shape) == 2:
            point_clouds = point_clouds.unsqueeze(0)
        
        for batch_idx in range(point_clouds.shape[0]):
            cur_pc = point_clouds[batch_idx]  # N x 3
            height = (cur_pc[:, 2] * 100).to(torch.int32)  # N
            height, _ = torch.sort(height, dim=-1)
            lowest = height[0] 
            statics_height = torch.bincount(height - lowest)  # N
            max_num_height = statics_height.argmax() * 0.01 + lowest * 0.01 
            cur_ground_mask = cur_pc[:, 2] < max_num_height + 0.2  # N
            ground_mask.append(cur_ground_mask)
            
        ground_mask = torch.stack(ground_mask)  # B x N
        return ground_mask 

    def init_bev(
        self,
        batch_size=1,
        channel=3,
        fill_value=255.0,
        device='cuda'
    ):
        return torch.ones([batch_size, *self.grid_shape, channel], dtype=torch.float32, device=device) * fill_value

    def get_binary_bev(
        self,
        point_clouds,
        *args,
        **kwargs,
    ) -> torch.Tensor :
        '''tranlate point clouds into binary bev map

        Args:
            point_clouds (torch.Tensor): B x N x (3 + C)

        Returns:
            bev_map (torch.Tensor): B x H x W x 3
        '''
        B, N, _ = point_clouds.shape

        # x, y rank
        rank_x, rank_y = self.ranks_from_coors(point_clouds)
        batch_id = torch.arange(0, B, dtype=torch.long).view(B, 1).expand(B, N)  # B x N
        
        bev_map = self.init_bev(B, channel=1, fill_value=0, device=point_clouds.device)
        bev_map[batch_id, rank_x, rank_y, :] = self.fill_value
        bev_map = bev_map.expand(B, *self.grid_shape, 3)  # B x H x W x 3
        
        return bev_map

    def get_intensity_bev(
        self,
        point_clouds,
        *args,
        **kwargs,
    ) -> torch.Tensor :
        '''translate point clouds into bev map, each grid is determined by the point cloud intensities

        Args:
            point_clouds (torch.Tensor): B x N x (3 + 1 + C)

        Returns:
            bev_map (torch.Tensor): B x H x W x 3
        '''
        B, N, _ = point_clouds.shape

        pc_intensity = point_clouds[:, :, 3:4]  # B x N x 1

        # x, y rank
        rank_x, rank_y = self.ranks_from_coors(point_clouds)
        batch_id = torch.arange(0, B, dtype=torch.long).view(B, 1).expand(B, N)  # B x N
        
        bev_map = self.init_bev(B, channel=1, fill_value=0, device=point_clouds.device)
        # regularize the pc_intensity to [0, 1]
        pc_intensity = pc_intensity.log()
        intensity_min, intensity_max = pc_intensity.min(), pc_intensity.max()
        pc_intensity = (pc_intensity - intensity_min) / (intensity_max - intensity_min)
        # map intensity to gray value
        fill_value = pc_intensity * 255.0
        bev_map[batch_id, rank_x, rank_y, :] = fill_value
        bev_map = bev_map.expand(B, *self.grid_shape, 3)  # B x H x W x 3

        return bev_map

    def get_image_rgb_bev(
        self,
        point_clouds,
        batch_imgs_list,
        batch_input_meta,
        *args,
        **kwargs,
    ) -> torch.Tensor :
        '''translate point clouds and corresponding images into bev maps,
            the color of each grid is determined by its projected image colors 

        Args:
            point_clouds (torch.Tensor): B x N x (3 + C)
            batch_imgs_list (List(torch.Tensor)): the list of images, each entry of shape num_views x 3 x H x W
            batch_input_meta (List[dict]): the input meta infos

        Returns:
            bev_map (torch.Tensor): B x H x W x 3
        '''
        mv_img_list = batch_imgs_list
        B, N, _ = point_clouds.shape

        # x, y rank
        pc_coors = point_clouds[:, :, :3]  # B x N x 3
        rank_x, rank_y = self.ranks_from_coors(point_clouds)
        batch_id = torch.arange(0, B, dtype=torch.long).view(B, 1).expand(B, N)  # B x N
        
        bev_map = self.init_bev(B, channel=3, fill_value=0, device=point_clouds.device)

        frame_volumes = list()

        for batch_idx, img_meta in enumerate(batch_input_meta):
            num_views = img_meta['num_views']
            mv_imgs = mv_img_list[batch_idx]  # num_views x 3 x H x W
            cur_points = pc_coors[batch_idx]  # N x 3
            volume_feats = list()

            for view_id in range(num_views):  # process each view
                # ipdb.set_trace()
                cur_img = mv_imgs[view_id : view_id + 1]  # 1 x 3 x H x W
                # 1. prepare lidar to each view projection mat
                lidar2cam = cur_points.new_tensor(img_meta['lidar2cam'][view_id])  # 4 x 4
                cam2img = cur_points.new_tensor(img_meta['ori_cam2img'][view_id])  # 4 x 4
                lidar2img = get_lidar2img(cam2img.double(), lidar2cam.double()).float()
                # 2. prepare img aug info
                if 'scale_factor' in img_meta:
                    img_scale_factor = img_meta['scale_factor']
                    if isinstance(img_scale_factor, list):
                        img_scale_factor = img_scale_factor[view_id]
                    if isinstance(img_scale_factor, np.ndarray) and len(img_meta['scale_factor']) >= 2:
                        img_scale_factor = (cur_points.new_tensor(img_scale_factor[:2]))
                    else:
                        img_scale_factor = (cur_points.new_tensor(img_scale_factor))
                else:
                    img_scale_factor = (1)
                if 'img_crop_offset' in img_meta:
                    img_crop_offset = cur_points.new_tensor(img_meta['img_crop_offset'][view_id])
                else:
                    img_crop_offset = 0
                img_flip = img_meta['flip'][view_id] if 'flip' in img_meta.keys() else False
                img_pad_shape = img_meta['pad_shape'][:2]
                img_shape = img_meta['img_shape'][:2]

                cur_feat, _ = point_sample(
                    img_meta=img_meta,
                    img_features=cur_img,
                    points=cur_points,
                    proj_mat=lidar2img,
                    coord_type='LIDAR',
                    img_scale_factor=img_scale_factor,
                    img_crop_offset=img_crop_offset,
                    img_flip=img_flip,
                    img_pad_shape=img_pad_shape,
                    img_shape=img_shape,
                    aligned=False,
                    valid_flag=True
                )  # N x C
                volume_feats.append(cur_feat)
            
            volume_feats = torch.stack(volume_feats)  # num_views x N x 3
            # TODO: is it ok to simply sum up features of all views?
            volume_feats = volume_feats.sum(0)  # N x 3

            frame_volumes.append(volume_feats)
        
        frame_volumes = torch.stack(frame_volumes)  # B x N x 3
        bev_map[batch_id, rank_x, rank_y, :] = frame_volumes

        return bev_map

    def get_intensity_rgb_bev(
        self,
        point_clouds: torch.Tensor,
        *args,
        **kwargs,
    ):
        B, N, _ = point_clouds.shape
        pc_intensity = point_clouds[:, :, 3]  # B x N

        # x, y rank
        rank_x, rank_y = self.ranks_from_coors(point_clouds)
        batch_id = torch.arange(0, B, dtype=torch.long).view(B, 1).expand(B, N)  # B x N

        bev_map = self.init_bev(B, channel=3, fill_value=0, device=point_clouds.device)

        # filter out the ground points
        if self.remove_ground_points:
            ground_mask = self.distinguish_ground(point_clouds)  # B x N

        # for ground points, assign gray color, for other points, assign colors according to intensity
        color_map = plt.get_cmap('rainbow')
        # regularize the pc_intensity to [0, 1]
        pc_intensity = pc_intensity.log()
        intensity_min, intensity_max = pc_intensity.min(), pc_intensity.max()
        pc_intensity = (pc_intensity - intensity_min) / (intensity_max - intensity_min)
        pc_intensity_np = pc_intensity.view(B * N).cpu().numpy()
        pc_colors = torch.from_numpy(color_map(pc_intensity_np) * 255)[:, :3].to(torch.float32) 
        pc_colors = pc_colors.view(B, N, 3).to(point_clouds.device)  # B x N x 3
        if self.remove_ground_points:
            pc_colors.masked_fill_(ground_mask.unsqueeze(-1), 0)  # fill black color

        bev_map[batch_id, rank_x, rank_y, :] = pc_colors
        return bev_map


@MODELS.register_module()
class BEVPostProcessor(BaseModule):
    SUPPORT_PROCESS = ['dilate', 'erode']

    def __init__(
        self, 
        process_list: List[dict] = None,
        channel_first = True,
        init_cfg = None
    ):
        super().__init__(init_cfg)

        self.process_list = process_list
        self.channel_first = channel_first
    
    def forward(self, bev_map: torch.Tensor):
        if self.process_list is None:
            return bev_map  # do nothing

        for process_cfg in self.process_list:
            process_type = process_cfg.get('type')
            if process_type in BEVPostProcessor.SUPPORT_PROCESS:
                bev_map = getattr(self, f'{process_type}_bev')(bev_map, process_cfg)
            else:
                raise NotImplementedError(f'Not supported bev post-process: {process_type}')
        
        return bev_map

    def dilate_bev(self, bev_map: torch.Tensor, dilate_config: dict):
        if not self.channel_first:
            bev_map = bev_map.permute([0, 3, 1, 2]).contiguous()  # B x C x H x W
        
        kernel_size = dilate_config.get('kernel_size')
        repeat = dilate_config.get('repeat', 1)
        pad_size = int((kernel_size - 1) / 2)

        dilate_op = nn.MaxPool2d(kernel_size, 1, pad_size)
        dilate_bev = bev_map
        for _ in range(repeat):
            dilate_bev = dilate_op(dilate_bev)

        if not self.channel_first:
            dilate_bev = dilate_bev.permute([0, 2, 3, 1]).contiguous()  # B x H x W x C

        return dilate_bev

    def erode_bev(self, bev_map: torch.Tensor, erode_config: dict):
        if not self.channel_first:
            bev_map = bev_map.permute([0, 3, 1, 2]).contiguous()  # B x C x H x W
        
        kernel_size = erode_config.get('kernel_size')
        pad_size = int((kernel_size - 1) / 2)

        erode_op = nn.MaxPool2d(kernel_size, 1, pad_size)
        erode_bev = - erode_op(- bev_map)

        if not self.channel_first:
            erode_bev = erode_bev.permute([0, 2, 3, 1]).contiguous()  # B x H x W x C

        return erode_bev
