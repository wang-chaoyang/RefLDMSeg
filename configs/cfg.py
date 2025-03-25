# dataset settings
dataset_type = 'mmseg.PairDataset'
num_workers = 4

crop_size = (256,256)
train_pipeline_fs = [
    dict(type='LoadImageFromFile'),
    dict(type='mmseg.LoadAnnotations'),
    dict(type='mmseg.FSTransform'),
    dict(
        type='RandomResize',
        scale=(512,256),
        ratio_range=(0.9, 1.1),
        keep_ratio=True),
    dict(type='mmseg.RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='mmseg.RandomFlip', prob=0.5),
    dict(type='mmseg.PhotoMetricDistortion'),
    dict(type='mmseg.PackSegInputs',meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label','cls_chosen'))
]

train_pipeline_vos = [
    dict(type='LoadImageFromFile'),
    dict(type='mmseg.LoadAnnotations'),
    dict(type='mmseg.VOSTransform'),
    dict(
        type='RandomResize',
        scale=(512,256),
        ratio_range=(0.9, 1.1),
        keep_ratio=True),
    dict(type='mmseg.RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='mmseg.PhotoMetricDistortion'),
    dict(type='mmseg.PackSegInputs',meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor','reduce_zero_label','cls_chosen'))
]

train_pipeline_multivos = [
    dict(type='LoadImageFromFile'),
    dict(type='mmseg.LoadAnnotations'),
    dict(type='mmseg.VOSMultiTransform'),
    dict(
        type='RandomResize',
        scale=(512,256),
        ratio_range=(0.9, 1.1),
        keep_ratio=True),
    dict(type='mmseg.RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='mmseg.PhotoMetricDistortion'),
    dict(type='mmseg.PackSegInputs',meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor','reduce_zero_label','cls_chosen'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size,keep_ratio=True),
    dict(type='mmseg.LoadAnnotations'),   # 
    dict(type='mmseg.FSTransform'),
    dict(type='mmseg.PackSegInputs',meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label','cls_chosen'))
]

supp_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmseg.LoadAnnotations'),
    dict(type='mmseg.FSTransform'),
    dict(type='Resize', scale=crop_size,keep_ratio=True),
    dict(type='mmseg.PackSegInputs',meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label','cls_chosen'))
]

test_pipeline_vos = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size,keep_ratio=True),
    dict(type='mmseg.LoadAnnotations'),   # 
    dict(type='mmseg.VOSTransform'),
    dict(type='mmseg.PackSegInputs',meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label','cls_chosen'))
]

supp_pipeline_vos = [
    dict(type='LoadImageFromFile'),
    dict(type='mmseg.LoadAnnotations'),
    dict(type='mmseg.VOSTransform'),
    dict(type='Resize', scale=crop_size,keep_ratio=True),
    dict(type='mmseg.PackSegInputs',meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label','cls_chosen'))
]


pascal_train=dict(
        type=dataset_type,
        data_root='datasets/pascal',
        meta_list='datasets/pascal/metas/meta_train.npy',
        supp_pipeline=train_pipeline_fs,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClassAug'),
        ann_file='metas/train.txt',
        pipeline=train_pipeline_fs)

coco_train=dict(
        type=dataset_type,
        data_root='datasets/coco20i',
        meta_list='datasets/coco20i/metas/meta_train.npy',
        supp_pipeline=train_pipeline_fs,
        data_prefix=dict(
            img_path='train2014', seg_map_path='annotations/train2014'),
        ann_file='metas/train.txt',
        pipeline=train_pipeline_fs)

davis_train=dict(
        type=dataset_type,
        data_root='datasets/davis16',
        meta_list='datasets/davis16/metas/480p/meta_train.npy',
        supp_pipeline=train_pipeline_vos,
        vos=True,
        data_prefix=dict(
            img_path='JPEGImages/480p', seg_map_path='Annotations/480p'),
        ann_file='metas/480p/train.txt',
        pipeline=train_pipeline_vos)

vspw_train=dict(
        type=dataset_type,
        data_root='datasets/vspw',
        meta_list='datasets/vspw/metas/meta_train.npy',
        supp_pipeline=train_pipeline_multivos,
        vos=True,
        data_prefix=dict(
            img_path='images', seg_map_path='masks'),
        ann_file='metas/train.txt',
        pipeline=train_pipeline_multivos)

train_dataloader = dict(
    batch_size=8,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='mmseg.WeightedRandomSampler',weight_list=[0.4,0.1,0.5,0.2],seed=123),
    dataset=dict(type='mmseg.ConcatDataset', datasets=[pascal_train,coco_train,davis_train,vspw_train]))

val_dataloader_coco = dict(
    batch_size=1,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root='datasets/coco20i',
        meta_list='datasets/coco20i/metas/meta_val.npy',
        supp_pipeline=supp_pipeline,
        test_mode=True,
        data_prefix=dict(
            img_path='val2014', seg_map_path='annotations/val2014'),
        ann_file='metas/val.txt',
        pipeline=test_pipeline))

val_dataloader_pascal = dict(
    batch_size=1,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root='datasets/pascal',
        meta_list='datasets/pascal/metas/meta_val.npy',
        supp_pipeline=supp_pipeline,
        test_mode=True,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClassAug'),
        ann_file='metas/val.txt',
        pipeline=test_pipeline))


data_preprocessor = dict(
    type='PairPreProcessor',
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    size=crop_size,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size=crop_size))

