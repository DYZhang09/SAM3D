import torch
import cv2
import numpy as np

def bev_visualize(bev_map: torch.Tensor, output_path):
    '''visualize a SINGLE bev map

    Args:
        bev_map (numpy.ndarray): H x W x 3
        output_path (str): indicates where to save vis result
    '''
    canvas = bev_map
    canvas = canvas.cpu().numpy()

    cv2.imwrite(output_path, canvas)


def minrect_vis(cv_rect_res, canvas, to_save=False, save_name=None):
    box = cv2.boxPoints(cv_rect_res) 
    box = np.int0(box)
    cv2.drawContours(canvas, [box], 0, (0, 255, 0), 2)
    
    if to_save:
        assert save_name is not None
        cv2.imwrite(save_name, canvas.astype(np.uint8))
