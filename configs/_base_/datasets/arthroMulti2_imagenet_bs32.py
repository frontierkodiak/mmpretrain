# dataset settings
dataset_type = 'MultiTaskDataset'
data_preprocessor = dict(
    #num_classes=1000,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackMultiTaskInputs', multi_task_fields=('gt_label', )),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackMultiTaskInputs', multi_task_fields=('gt_label', )),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=0,
    dataset=dict(
        type=dataset_type,
        data_root='/peach/NA_arthropoda_min180all_cap1500_Jul23/224_95q/train/',
        ann_file='/peach/NA_arthropoda_min180all_cap1500_Jul23/224_95q/annotation/verified_train_trimmed_1k.json',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=0,
    dataset=dict(
        type=dataset_type,
        data_root='/peach/NA_arthropoda_min180all_cap1500_Jul23/224_95q/val/',
        ann_file='/peach/NA_arthropoda_min180all_cap1500_Jul23/224_95q/annotation/verified_val_trimmed_1k.json',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(
    type='MultiTasksMetric',
    task_metrics={
        'L10': [dict(type='Accuracy', topk=(1))],
        'L20': [dict(type='Accuracy', topk=(1))]
    })
# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
