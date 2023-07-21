# Save as `configs/resnet/multi-task-demo.py`
_base_ = ['../_base_/schedules/cifar10_bs128.py', '../_base_/default_runtime.py']

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='ResNet_CIFAR', depth=18),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiTaskClsHead',                                    # <- Head config, depends on #675
        sub_heads={
            'L10': dict(type='LinearClsHead', num_classes=871),
            'L20': dict(type='LinearClsHead', num_classes=407),
            'L30': dict(type='LinearClsHead', num_classes=94),
            'L40': dict(type='LinearClsHead', num_classes=25),
        },
        common_cfg=dict(
            in_channels=512,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        ),
    ),
)

# dataset settings
dataset_type = 'MultiTaskDataset'
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='FormatMultiTaskLabels'),                             # <- Use this to replace `ToTensor`.
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file='/peach/NA_aves_min500rg_cap2500/224_90q/train/labels.json',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='/peach/NA_aves_min500rg_cap2500/224_90q/val/labels.json',
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        ann_file='/peach/NA_aves_min500rg_cap2500/224_90q/val/labels.json',
        pipeline=test_pipeline,
        test_mode=True))

evaluation = dict(metric_options={
    'L10': dict(topk=(1, 3)),                # <- Specify different metric options for different tasks.
    'L20': dict(topk=(1, 3)),
    'L30': dict(topk=(1, 3)),
    'L40': dict(topk=(1, 3)),
})