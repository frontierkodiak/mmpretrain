_base_ = [
    '../_base_/models/mae_vit-base-p16.py',
    '../_base_/datasets/imagenet_bs512_mae.py',
    '../_base_/default_runtime.py',
]

### NOTE: Base: mae_vit-base-p16_8xb512-amp-coslr-300e_in1k

load_from = "/modelZoo/mae_vit-base-p16_8xb512-coslr-300e-fp16_in1k_20220829-c2cf66ba.pth"

# model settings 
model = dict(
    type='MAE',
    backbone=dict(type='MAEViT', arch='b', patch_size=16, mask_ratio=0.75),
    neck=dict(
        type='MAEPretrainDecoder',
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
    ),
    # head=dict( # Single-task head
    #     type='MAEPretrainHead',
    #     norm_pix=True,
    #     patch_size=16,
    #     loss=dict(type='PixelReconstructionLoss', criterion='L2')),
    head=dict( # Multi-task head
        type='MultiTaskHead',
        task_heads={
            'L10': dict(type='LinearClsHead', num_classes=871, in_channels=512, loss=dict(type='CrossEntropyLoss', loss_weight=1.0)),
            'L20': dict(type='LinearClsHead', num_classes=407, in_channels=512, loss=dict(type='CrossEntropyLoss', loss_weight=1.0)),
            'L30': dict(type='LinearClsHead', num_classes=94, in_channels=512, loss=dict(type='CrossEntropyLoss', loss_weight=1.0)),
            'L40': dict(type='LinearClsHead', num_classes=25, in_channels=512, loss=dict(type='CrossEntropyLoss', loss_weight=1.0)),
        },
        common_cfg=dict(
            in_channels=512,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        ),
    ),
    init_cfg=[
        dict(type='Xavier', layer='Linear', distribution='uniform'),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ])

# optimizer wrapper
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW',
        lr=1.5e-4 * 4096 / 256,
        betas=(0.9, 0.95),
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=260,
        by_epoch=True,
        begin=40,
        end=300,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300, val_interval=3)

# Default hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),  # Update the time spent during iteration into message hub
    logger=dict(type='LoggerHook', interval=250),  # Collect logs from different components of Runner and write them to terminal, JSON file, tensorboard and wandb .etc
    param_scheduler=dict(type='ParamSchedulerHook'), # update some hyper-parameters of optimizer
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto', max_keep_ckpts=3) # Save checkpoints periodically
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
                'project': 'Mb1A2',
                'group': 'aves_multi',
                'name': 'E21.0.300e.mae_base.aves_multi'
            })
    ]
)

randomness = dict(seed=0, diff_rank_seed=True)

# auto resume
resume = True

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=4096)
