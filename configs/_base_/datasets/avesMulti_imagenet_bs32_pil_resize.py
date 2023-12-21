# dataset settings
dataset_type = 'MultiTaskDataset'

data_preprocessor = dict(mean=[123.675, 116.28, 103.53],std=[58.395, 57.12, 57.375], to_rgb=True,)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackMultiTaskInputs', multi_task_fields=('gt_label', )),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackMultiTaskInputs', multi_task_fields=('gt_label', )),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=2, # NOTE: Change back to 1 if memory is overloaded
    dataset=dict(
        type=dataset_type,
        ann_file='/peach/NA_aves_min500rg_cap2500/224_90q/train/verified_labels.json',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=2, # NOTE: Change back to 1 if memory is overloaded
    dataset=dict(
        type=dataset_type,
        ann_file='/peach/NA_aves_min500rg_cap2500/224_90q/val/verified_labels.json',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(
    type='MultiTasksMetric',
    task_metrics={
        'L10': [dict(type='Accuracy', topk=(1,3))],
        'L20': [dict(type='Accuracy', topk=(1,3))],
    })

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
