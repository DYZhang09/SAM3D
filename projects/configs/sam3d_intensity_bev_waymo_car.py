_base_ = [
    'mmdet3d::_base_/datasets/waymoD5-3d-3class.py',
    'mmdet3d::_base_/schedules/cyclic-40e.py', 
    'mmdet3d::_base_/default_runtime.py'
]
custom_imports = dict(
    imports=['projects.core'], allow_failed_imports=False)

voxel_size = [0.1, 0.1, 6]
point_cloud_range = [-30.0, -30.0, -2, 30.0, 30.0, 4]
class_names = ['Car']
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=True, use_camera=True)

model = dict(
    type='SAMDet3D',
    bev_mapper = dict(
        type='BEVMapper',
        mode='intensity_rgb',
        point_cloud_range=point_cloud_range,
        grid_xy_size=voxel_size[:2],
        channel_first=True,
        remove_ground_points=True,
    ),
    bev_postprocess = dict(
        type='BEVPostProcessor',
        process_list= [
            dict(type='dilate', kernel_size=3, repeat=1),
        ],
        channel_first=True,
    ),
    mask_generator = dict(
        type='MaskAutoGenerator',
        sam_type='default',
        sam_ckpt_path='projects/pretrain_weights/sam_vit_h_4b8939.pth',
    ),
    mask_postprocess = dict(
        type='MaskPostProcessor',
        process_list = [
            dict(type='get_rbox'),
            dict(type='remove_masks_by_rbox_area', min_threshold=200, max_threshold=5000),
            dict(type='remove_masks_by_aspect_ratio', min_threshold=1.5 ,max_threshold=4)
        ]
    ),
    vis=False  # if set to True, it will save the bev visualization in `mask_vis` folder
)

# dataset settings
eval_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=6, use_dim=5),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),  # only use for visualization
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        modality=input_modality,
        data_prefix=dict(
            pts='training/velodyne',
            CAM_FRONT='training/image_0',
            CAM_FRONT_RIGHT='training/image_1',
            CAM_FRONT_LEFT='training/image_2',
            CAM_SIDE_RIGHT='training/image_3',
            CAM_SIDE_LEFT='training/image_4',
        ),
    ))

test_dataloader = dict(
    dataset=dict(
        pipeline=eval_pipeline, 
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        data_prefix=dict(
            pts='training/velodyne',
            CAM_FRONT='training/image_0',
            CAM_FRONT_RIGHT='training/image_1',
            CAM_FRONT_LEFT='training/image_2',
            CAM_SIDE_RIGHT='training/image_3',
            CAM_SIDE_LEFT='training/image_4',
        ),
    ),   
)
test_cfg = dict()

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')