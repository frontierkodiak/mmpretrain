_base_ = [
    '../_base_/datasets/arthroMulti_bs64_swin_224_augmented.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

train_dataloader = dict(batch_size=194)
val_dataloader = dict(batch_size=170)
test_dataloader = val_dataloader

# model settings 
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='base',
        img_size=224,
        patch_size=16,
        drop_path_rate=0.1,
        out_type='avg_featmap',
        final_norm=False,
        init_cfg=dict(type='Pretrained', checkpoint="/modelZoo/mae_vit-base-p16_8xb512-coslr-300e-fp16_in1k_20220829-c2cf66ba.pth")),
    neck=None,
    head=dict(  # Multi-task head
        type='MultiTaskHead',
        task_heads={
            'L10': dict(type='LinearClsHead', num_classes=6315, in_channels=768, 
                        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'), 
                        init_cfg=[dict(type='TruncNormal', layer='Linear', std=2e-5)]),
            'L20': dict(type='LinearClsHead', num_classes=4198, in_channels=768, 
                        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'), 
                        init_cfg=[dict(type='TruncNormal', layer='Linear', std=2e-5)]),
            'L30': dict(type='LinearClsHead', num_classes=702, in_channels=768, 
                        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
                        init_cfg=[dict(type='TruncNormal', layer='Linear', std=2e-5)]),
            'L40': dict(type='LinearClsHead', num_classes=68, in_channels=768, 
                        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
                        init_cfg=[dict(type='TruncNormal', layer='Linear', std=2e-5)]),
            'L50': dict(type='LinearClsHead', num_classes=12, in_channels=768, 
                        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
                        init_cfg=[dict(type='TruncNormal', layer='Linear', std=2e-5)]),
        },
    ),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]))


# optimizer wrapper
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=2e-3, weight_decay=0.05, betas=(0.9, 0.999)),
    constructor='LearningRateDecayOptimWrapperConstructor',
    paramwise_cfg=dict(
        layer_decay_rate=0.65,
        custom_keys={
            '.ln': dict(decay_mult=0.0),
            '.bias': dict(decay_mult=0.0),
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=95,
        by_epoch=True,
        begin=5,
        end=100,
        eta_min=1e-6,
        convert_to_iter_based=True)
]

# Default hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),  # Update the time spent during iteration into message hub
    logger=dict(type='LoggerHook', interval=100),  # Collect logs from different components of Runner and write them to terminal, JSON file, tensorboard and wandb .etc
    param_scheduler=dict(type='ParamSchedulerHook'), # update some hyper-parameters of optimizer
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto', max_keep_ckpts=20) # Save checkpoints periodically
) 

# Visualization settings
# https://mmengine.readthedocs.io/en/latest/api/visualization.html
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(type='WandbVisBackend',
            init_kwargs={
                'project': 'Mi1A2',
                'group': 'arthro_multi',
                'name': 'E21.2.vit-base-p16_8xb128-coslr-100e_arthroMulti'
            })
    ]
)

train_cfg = dict(by_epoch=True, max_epochs=100)

randomness = dict(seed=0, diff_rank_seed=True)
