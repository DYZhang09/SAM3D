import ipdb
import os
import time
import numpy as np

import cv2
import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.models.detectors.base import Base3DDetector
from .vis_utils import bev_visualize, minrect_vis

@MODELS.register_module()
class SAMDet3D(Base3DDetector):
    def __init__(
        self,
        bev_mapper,
        mask_generator,
        bev_postprocess=None,
        mask_postprocess=None,
        init_cfg=None,
        data_preprocessor=None,
        **kwargs,
    ):
        super(SAMDet3D, self).__init__(init_cfg=init_cfg, data_preprocessor=data_preprocessor)
        self.bev_mapper = MODELS.build(bev_mapper)
        self.mask_generator = MODELS.build(mask_generator)
        if bev_postprocess is not None:
            self.bev_postprocess = MODELS.build(bev_postprocess)
        else:
            self.bev_postprocess = None
        if mask_postprocess is not None:
            self.mask_postprocess = MODELS.build(mask_postprocess)
        else:
            self.mask_postprocess = None

        # if need to visualize
        self.vis = kwargs.get('vis', False)
    
    def predict(self, batch_inputs_dict: dict, batch_data_samples):
        ''' inference

        Args:
            batch_inputs_dict (dict): inputs
            batch_data_samples (List(Det3DDataSample)): contains metainfos
                'gt_instances_3d' (InstanceData): the gt 3D objects infos
                    'bboxes_3d' (Instance3DBoxes): the gt 3D bboxes
        '''
        batch_input_metas = [item.metainfo for item in batch_data_samples]

        feat = self.bev_mapper(batch_inputs_dict, batch_input_metas)

        if self.bev_postprocess is not None:
            feat = self.bev_postprocess(feat)

        # masks_meta = self.generate_sam_mask(feat)
        masks_meta = self.mask_generator(feat)
        if self.mask_postprocess is not None:
            masks_meta = self.mask_postprocess(masks_meta)

        # visualization
        if self.vis:
            full_lidar_path = batch_input_metas[0]['lidar_path']
            abs_folder, full_lidar_file = os.path.split(full_lidar_path)
            print(full_lidar_file)
            lidar_name, _ = os.path.splitext(full_lidar_file)
            file_path = os.path.join('./mask_vis', lidar_name)
            os.makedirs(file_path, exist_ok=True)

            # bev vis
            bev_image = feat[0].permute(1, 2, 0).contiguous()
            bev_visualize(bev_image, os.path.join(file_path, 'bev.png'))
            # mask vis
            for i, mask_data in enumerate(masks_meta):
                file_name = f"{i}.png"
                mask = mask_data['segmentation']
                os.makedirs(file_path, exist_ok=True)
                cv2.imwrite(os.path.join(file_path, file_name), mask * 255)
            # img vis
            mv_imgs = batch_inputs_dict.get('img', None)
            if mv_imgs is not None:
                mv_imgs = mv_imgs[0]  # num_views x 3 x H x W
                num_views = mv_imgs.shape[0]
                for view_id in range(num_views):
                    img = torch.permute(mv_imgs[view_id], [1, 2, 0]).contiguous()  # H x W x 3
                    cv2.imwrite(os.path.join(file_path, f'{view_id}_view.png'), img.cpu().numpy())

        results_list = self.mask2results(batch_inputs_dict, masks_meta, batch_input_metas, bev_image.cpu().numpy() if self.vis else None)
        
        return self.add_pred_to_datasample(batch_data_samples, results_list)

    def mask2results(self, batch_inputs_dict, masks_meta, batch_input_metas, bev_image=None):
        '''translate masks into bboxes, ONLY SUPPORT SINGLE SAMPLE

        Args:
            masks_meta list(dict(str, any)): A list over records for masks. Each record is
                a dict containing the following keys:
                segmentation (dict(str, any) or np.ndarray): The mask. If
                    output_mode='binary_mask', is an array of shape HW. Otherwise,
                    is a dictionary containing the RLE.
                bbox (list(float)): The box around the mask, in XYWH format.
                area (int): The area in pixels of the mask.
                predicted_iou (float): The model's own prediction of the mask's
                    quality. This is filtered by the pred_iou_thresh parameter.
                point_coords (list(list(float))): The point coordinates input
                    to the model to generate this mask.
                stability_score (float): A measure of the mask's quality. This
                    is filtered on using the stability_score_thresh parameter.
                crop_box (list(float)): The crop of the image used to generate
                    the mask, given in XYWH format.
            batch_input_metas (_type_): _description_

        Returns:
            _type_: _description_
        '''
        r_bboxes = []
        bboxes_scores = []
        
        vis_canvas = bev_image
        pc = batch_inputs_dict.get('points')  # list, each entry is of shape N x 3
        pc = pc[0]
        # generate 3D bbox
        for i, mask_info in enumerate(masks_meta):
            if 'rbox' not in mask_info:
                raise KeyError(f'Please use \'get_rbox\' in MaskPostProcessor first!')
            rect_res = mask_info['rbox']

            # 1.from pixel center to lidar center
            (cx, cy), (w, h), a = rect_res
            center_x, center_y = self.bev_mapper.coors_from_ranks(cy, cx)

            # 2.for orientation, need to translate from opencv definition (ver. > 4.5.0) to mmdet3d LiDAR definition
            a_rad = a * np.pi / 180.0 
            theta = np.pi / 2 - a_rad 
            dx = w * self.bev_mapper.grid_size[0]
            dy = h * self.bev_mapper.grid_size[1]

            # 3.estimate the gravity center and height
            fake_bboxes = batch_input_metas[0]['box_type_3d'](
                torch.tensor([[center_x, center_y, -100, dx, dy, 1000, theta]], device=pc.device), 7
            )
            inside_mask = fake_bboxes.points_in_boxes_all(pc).to(torch.bool).squeeze(-1)  # N
            if inside_mask.sum() > 0:
                inside_points = pc[inside_mask]  # M x 3
                inside_z = inside_points[:, 2]  
                z_max, z_min = inside_z.max().item(), inside_z.min().item()
                dz = z_max - z_min
                # Note: bottom center here
                center_z = z_min
                
                if dz > 0.8:
                    r_bboxes.append(np.array([center_x, center_y, center_z, dx, dy, dz, theta]))

                    score = mask_info['predicted_iou']
                    bboxes_scores.append(score)
            
            # visualization
            if self.vis:  
                if vis_canvas is None:  # create a new canvas
                    H, W = mask_info['segmentation'].shape
                    vis_canvas = np.zeros([H, W, 3])
                to_save = i == len(masks_meta) - 1
                if to_save:
                    full_lidar_path = batch_input_metas[0]['lidar_path']
                    _, full_lidar_file = os.path.split(full_lidar_path)
                    lidar_name, _ = os.path.splitext(full_lidar_file)
                    file_path = os.path.join('./mask_vis', lidar_name)
                    os.makedirs(file_path, exist_ok=True)
                    save_name = os.path.join(file_path, 'bev_box.png')
                minrect_vis(rect_res, vis_canvas, to_save, save_name if to_save else None)
            
        ret_list = list()
        temp_instances = InstanceData()
        if len(r_bboxes) == 0:
            r_bboxes.append(np.zeros(7))
            bboxes_scores.append(0)
        bboxes = batch_input_metas[0]['box_type_3d'](torch.from_numpy(np.array(r_bboxes)), 7)
        scores = torch.from_numpy(np.array(bboxes_scores)).to(device=bboxes.device)
        labels = torch.zeros_like(scores, dtype=torch.int)
        temp_instances.bboxes_3d = bboxes
        temp_instances.scores_3d = scores
        temp_instances.labels_3d = labels
        ret_list.append(temp_instances)
        return ret_list

    def loss(self, batch_inputs_dict: dict, batch_data_samples):
        ipdb.set_trace()
        pass

    def _forward(self, batch_inputs, batch_data_samples):
        ipdb.set_trace()
        pass

    def extract_feat(self, batch_inputs: Tensor):
        pass
    

####################### used for debug ##########################
from mmengine.model import BaseModule
from mmdet3d.structures.bbox_3d.base_box3d import BaseInstance3DBoxes

class BoxFilter(BaseModule):
    def __init__(
        self, 
        bev_dist_threshold,
        bbox_type='lidar',
        init_cfg=None
    ):
        '''Hungary Matcher used for bipartite matching

        Args:
            cost_class (_type_): _description_
            cost_objectness (_type_): _description_
            cost_corner ():
            cost_center (_type_): _description_
            init_cfg (_type_, optional): _description_. Defaults to None.
        '''
        super().__init__(init_cfg)
        self.dist_threshold = bev_dist_threshold
        self.bbox_type = bbox_type.lower()

    @torch.no_grad()
    def forward(
        self, 
        pred_bboxes_3d: BaseInstance3DBoxes,
        gt_bboxes_3d: BaseInstance3DBoxes, 
    ):
        '''Forward pass, do hungary matching
            NOTE: only used for SINGLE sample

        Args:
            cls_pred_scores (torch.Tensor): the scores(probabilities) of class branch with shape of (num_query, num_classes + 1)
            bbox3d_pred (torch.Tensor): the final prediction of heads with shape of (num_query, 7)
            gt_bboxes_3d (BaseInstance3DBoxes): ground-truth bboxes with shape of (#gt, 7)
            gt_labels_3d (torch.Tensor): ground-truth labels with shape of (#gt)
            gt_num (int): the actual gt num of each sample
            img_metas (dict): the list of image meta info
        '''

        pred_bev_center = pred_bboxes_3d.gravity_center[..., :2]
        tgt_bev_center = gt_bboxes_3d.gravity_center[..., :2]
        bbox_mat = torch.cdist(pred_bev_center, tgt_bev_center, p=2)  # #pred x #gt

        min_dist, per_prop_gt_inds = torch.min(bbox_mat, dim=-1)  # #pred
        proposal_matched_mask = min_dist <= self.dist_threshold

        return {
            "per_prop_gt_inds": per_prop_gt_inds,
            "proposal_matched_mask": proposal_matched_mask,
        }