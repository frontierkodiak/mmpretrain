_base_ = [
    '../_base_/models/resnext50_32x4d_avesMulti.py',
    '../_base_/datasets/avesMulti_bs64_swin_224_wsl_augmented.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

train_dataloader = dict(batch_size=64)
val_dataloader = dict(batch_size=64)
test_dataloader = val_dataloader

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNeXt',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        groups=32,
        width_per_group=4,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint="~/local-data/modelZoo/mmpretrain/resnext50_32x4d_b32x8_imagenet_20210429-56066e27.pth")),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(  # Multi-task head
        type='MultiTaskHead',
        task_heads={
            'L10': dict(type='LinearClsHead', num_classes=871, in_channels=2048, 
                        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'), 
                        init_cfg=[dict(type='TruncNormal', layer='Linear', std=2e-5)]),
            'L20': dict(type='LinearClsHead', num_classes=407, in_channels=2048, 
                        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'), 
                        init_cfg=[dict(type='TruncNormal', layer='Linear', std=2e-5)]),
            'L30': dict(type='LinearClsHead', num_classes=94, in_channels=2048, 
                        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
                        init_cfg=[dict(type='TruncNormal', layer='Linear', std=2e-5)]),
            'L40': dict(type='LinearClsHead', num_classes=25, in_channels=2048, 
                        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
                        init_cfg=[dict(type='TruncNormal', layer='Linear', std=2e-5)]),
        },
    ))












# Default hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),  # Update the time spent during iteration into message hub
    logger=dict(type='LoggerHook', interval=100),  # Collect logs from different components of Runner and write them to terminal, JSON file, tensorboard and wandb .etc
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
        # dict(type='WandbVisBackend',
        #     init_kwargs={
        #         'project': 'Mb1A2',
        #         'group': 'aves_multi',
        #         'name': 'E22.0.resnext50-32x4d_8xb32_avesMulti'
        #     })
    ]
)


train_cfg = dict(by_epoch=True, max_epochs=100)
    # augments=[
    #     dict(type='Mixup', alpha=0.8),
    #     dict(type='CutMix', alpha=1.0)])

randomness = dict(seed=0, diff_rank_seed=True)