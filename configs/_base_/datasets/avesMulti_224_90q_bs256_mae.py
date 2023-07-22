# dataset settings
dataset_type = 'MultiTaskDataset'
data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        crop_ratio_range=(0.2, 1.0),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackInputs')
]

train_dataloader = dict(
    batch_size=256,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        ann_file='/peach/NA_aves_min500rg_cap2500/224_90q/train/labels.json',
        split='train',
        pipeline=train_pipeline))

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='PackInputs')
]

val_dataloader = dict(
    batch_size=256,
    num_workers=8,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        ann_file='/peach/NA_aves_min500rg_cap2500/224_90q/val/labels.json',
        split='val',
        pipeline=val_pipeline))

test_pipeline = val_pipeline
test_dataloader = val_dataloader
