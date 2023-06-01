from typing import List, Tuple
import cv2
import numpy as np
from segment_anything.modeling import Sam  
import torch
import torch.nn.functional as F
from torchvision.ops.boxes import batched_nms

from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator 
from segment_anything.utils.amg import MaskData, batch_iterator, uncrop_boxes_xyxy, uncrop_points 


@MODELS.register_module()
class MaskAutoGenerator(BaseModule):
    def __init__(
        self, 
        sam_type,
        sam_ckpt_path,
        init_cfg = None,
        *args,
        **kwargs,
    ):
        super().__init__(init_cfg)

        # build SAM segmentor
        self.sam = sam_model_registry[sam_type](checkpoint=sam_ckpt_path).to(device='cuda')
        self.mask_generator = FastSamAutomaticMaskGenerator(self.sam)

    def forward(self, feat: torch.Tensor):
        '''generate mask using SAM for bev map

        Args:
            bev_feat (torch.Tensor): the bev map of shape (1, C, H, W)

        Returns:
            masks: List[List[dict]]
        '''
        # TODO: batchify
        cur_bev = feat[0]  # 3 x H x W
        cur_bev = torch.permute(cur_bev, [1, 2, 0]).contiguous()  # H x W x 3
        # generate mask, SAM currently requires np.ndarray with dtype np.uint8
        cur_bev = cur_bev.cpu().numpy().astype(np.uint8)
        cur_mask = self.mask_generator.generate(cur_bev)
        return cur_mask


class FastSamAutomaticMaskGenerator(SamAutomaticMaskGenerator):
    def __init__(
        self, 
        model: Sam, 
        points_per_side = 32, 
        points_per_batch: int = 64, 
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95, 
        stability_score_offset: float = 1, 
        box_nms_thresh: float = 0.7, 
        crop_n_layers: int = 0, 
        crop_nms_thresh: float = 0.7, 
        crop_overlap_ratio: float = 512 / 1500, 
        crop_n_points_downscale_factor: int = 1, 
        point_grids = None, 
        min_mask_region_area: int = 0, 
        output_mode: str = "binary_mask",
        receptive_size: tuple = (3, 3)
    ) -> None:
        super().__init__(model, points_per_side, points_per_batch, pred_iou_thresh, stability_score_thresh, stability_score_offset, box_nms_thresh, crop_n_layers, crop_nms_thresh, crop_overlap_ratio, crop_n_points_downscale_factor, point_grids, min_mask_region_area, output_mode)
        self.perceptive_size = receptive_size
        if isinstance(self.perceptive_size, tuple):
            self.pad_size = [int((kernel_size - 1) / 2) for kernel_size in self.perceptive_size]
        else:
            self.pad_size = int((self.perceptive_size - 1) / 2)

    @torch.no_grad()
    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        self.predictor.set_image(cropped_im)

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale   # total_num_points x 2, float64

        # filter point prompts of backgrounds via max pooling
        cropped_im_tensor = torch.from_numpy(cropped_im).to(torch.float64).to('cuda').permute(2, 0, 1).contiguous()  # C x H x W
        query_grids = torch.from_numpy(points_for_image).to('cuda')  
        coor_x, coor_y = torch.split(query_grids, 1, dim=1)  # each is Nx1
        h, w = cropped_im_size
        norm_coor_y = coor_y / h * 2 - 1
        norm_coor_x = coor_x / w * 2 - 1
        query_grids = torch.cat([norm_coor_x, norm_coor_y],
                        dim=1).unsqueeze(0).unsqueeze(0)  # Nx2 -> 1x1xNx2

        activation = F.max_pool2d(cropped_im_tensor, kernel_size=self.perceptive_size, stride=1, padding=self.pad_size)  # C x H x W
        activation = F.grid_sample(activation.unsqueeze(0), query_grids).squeeze().t()  # total_num_points x C
        activation = torch.sum(activation, dim=-1)  # N
        fg_mask = (activation > 0).cpu().numpy()
        points_for_image = points_for_image[fg_mask]

        # Generate masks for this crop in batches
        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(points, cropped_im_size, crop_box, orig_size)
            data.cat(batch_data)
            del batch_data
        self.predictor.reset_image()

        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        return data
    


@MODELS.register_module()
class MaskPostProcessor(BaseModule):
    SUPPORT_PROCESS = [
        'remove_masks_by_rbox_area', 'remove_masks_by_predicted_iou',
        'get_rbox', 'remove_masks_by_aspect_ratio'
    ]

    def __init__(
        self, 
        process_list: List[dict] = None,
        init_cfg = None,
    ):
        super().__init__(init_cfg)

        self.process_list = process_list

    def forward(self, mask_metainfos: List[dict]):
        if self.process_list is None:
            return mask_metainfos

        for process_cfg in self.process_list:
            process_type = process_cfg.get('type')
            if process_type in MaskPostProcessor.SUPPORT_PROCESS:
                mask_metainfos = getattr(self, process_type)(
                    mask_metainfos,
                    **process_cfg
                )
            else:
                raise NotImplementedError(f'Not supported mask post-process: {process_type}')

        return mask_metainfos
            
    def remove_masks_by_rbox_area(
        self,
        mask_metainfos,
        min_threshold = None,
        max_threshold = None,
        **kwargs
    ):
        return self.__remove_masks_by_attr(mask_metainfos, 'rbox_area', min_threshold, max_threshold, **kwargs)

    def remove_masks_by_predicted_iou(
        self,
        mask_metainfos,
        min_threshold = None,
        max_threshold = None,
        **kwargs
    ):
        return self.__remove_masks_by_attr(mask_metainfos, 'predicted_iou', min_threshold, max_threshold, **kwargs)

    def get_rbox(
        self,
        mask_metainfos: List[dict],
        **kwargs
    ):
        for mask_info in mask_metainfos:
            mask = mask_info['segmentation']  # H x W
            y, x = np.nonzero(mask)
            fg_points = np.stack([x, y], axis=-1)
            rect_res = cv2.minAreaRect(fg_points)
            mask_info['rbox'] = rect_res

            # calculate aspect ratio (w : h)
            (_, _), (w, h), _ = rect_res
            if min(w, h) < 1:
                mask_info['aspect_ratio'] = 1e10
            else:
                mask_info['aspect_ratio'] = max(w, h) / min(w, h)
            mask_info['rbox_area'] = w * h

        return mask_metainfos

    def remove_masks_by_aspect_ratio(
        self,
        mask_metainfos: List[dict],
        min_threshold = None,
        max_threshold = None,
        **kwargs,
    ):
        if len(mask_metainfos) == 0:
            return mask_metainfos
        if 'rbox' not in mask_metainfos[0]:
            raise KeyError(f'Please use \'get_rbox\' to get aspect ratio info.')
        return self.__remove_masks_by_attr(mask_metainfos, 'aspect_ratio', min_threshold, max_threshold, **kwargs)

    def __remove_masks_by_attr(
        self,
        mask_metainfos: List[dict],
        attr,
        min_threshold = None,
        max_threshold = None,
        **kwargs
    ):
        new_metainfos = list()
        for mask_info in mask_metainfos:
            mask_attr = mask_info.get(attr)
            
            if min_threshold is not None and mask_attr < min_threshold:
                continue
            if max_threshold is not None and mask_attr > max_threshold:
                continue
            new_metainfos.append(mask_info)

        return new_metainfos
