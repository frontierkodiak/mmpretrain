# dataset settings
dataset_type = 'MultiTaskDataset'
# data_preprocessor = dict(
#     # num_classes=1000, # CLARIFY: What do I need to modify here?
#     # RGB format normalization parameters
#     mean=[123.675, 116.28, 103.53],
#     std=[58.395, 57.12, 57.375],
#     # convert image from BGR to RGB
#     to_rgb=True,
# )

mean=[123.675, 116.28, 103.53]
std=[58.395, 57.12, 57.375]

bgr_mean = mean[::-1]
bgr_std = std[::-1]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(type='PackMultiTaskInputs', multi_task_fields=('gt_label', )),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackMultiTaskInputs', multi_task_fields=('gt_label', )),
]

train_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        ann_file='~/local-data/NA_aves_min500rg_cap2500/224_93q/train/verified_labels_wsl.json',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        ann_file='~/local-data/NA_aves_min500rg_cap2500/224_93q/val/verified_labels_wsl.json',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(
    type='MultiTasksMetric',
    task_metrics={
        'L10': [dict(type='Accuracy', topk=(1, 3))],
        'L20': [dict(type='Accuracy', topk=(1, 3))],
        'L30': [dict(type='Accuracy', topk=(1, 3))],
        'L40': [dict(type='Accuracy', topk=(1, 3))]
    })
# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
