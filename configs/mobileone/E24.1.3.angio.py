_base_ = [
    '../_base_/models/mobileone/mobileone_s2_angioMulti2_raw.py',
    '../_base_/datasets/angioMulti2_imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs256_coslr_coswd_300e.py',
    '../_base_/default_runtime.py'
]

# schedule settings
# optim_wrapper = dict(paramwise_cfg=dict(norm_decay_mult=0.))
optim_wrapper = dict(
    paramwise_cfg=dict(norm_decay_mult=0.),
    optimizer=dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.0001))


train_dataloader = dict(batch_size=900)
val_dataloader = dict(batch_size=700)
test_dataloader = dict(batch_size=700)

bgr_mean = _base_.data_preprocessor['mean'][::-1]
base_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=7,
        magnitude_std=0.5,
        hparams=dict(pad_val=[round(x) for x in bgr_mean])),
    dict(type='PackMultiTaskInputs', multi_task_fields=('gt_label', )),
]

import copy  # noqa: E402

# modify start epoch RandomResizedCrop.scale to 160
# and RA.magnitude_level * 0.3
train_pipeline_1e = copy.deepcopy(base_train_pipeline)
train_pipeline_1e[1]['scale'] = 160
train_pipeline_1e[3]['magnitude_level'] *= 0.3
_base_.train_dataloader.dataset.pipeline = train_pipeline_1e

# modify 137 epoch's RandomResizedCrop.scale to 192
# and RA.magnitude_level * 0.7
train_pipeline_37e = copy.deepcopy(base_train_pipeline)
train_pipeline_37e[1]['scale'] = 192
train_pipeline_37e[3]['magnitude_level'] *= 0.7

# modify 112 epoch's RandomResizedCrop.scale to 224
# and RA.magnitude_level * 1.0
train_pipeline_112e = copy.deepcopy(base_train_pipeline)
train_pipeline_112e[1]['scale'] = 224
train_pipeline_112e[3]['magnitude_level'] *= 1.0

custom_hooks = [
    dict(
        type='SwitchRecipeHook',
        schedule=[
            dict(action_epoch=37, pipeline=train_pipeline_37e),
            dict(action_epoch=112, pipeline=train_pipeline_112e),
        ]),
    dict(
        type='EMAHook',
        momentum=5e-4,
        priority='ABOVE_NORMAL',
        update_buffers=True)
]

# Default hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),  # Update the time spent during iteration into message hub
    logger=dict(type='LoggerHook', interval=100),  # Collect logs from different components of Runner and write them to terminal, JSON file, tensorboard and wandb .etc
    param_scheduler=dict(type='ParamSchedulerHook'), # update some hyper-parameters of optimizer
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto', max_keep_ckpts=10) # Save checkpoints periodically
) 


# Visualization settings
# https://mmengine.readthedocs.io/en/latest/api/visualization.html
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend',
            init_kwargs={
                'project': 'Mx1A2-angio',
                'group': 'E24.mobileone-s2',
                'name': 'E24.1.3'
            })
    ]
)