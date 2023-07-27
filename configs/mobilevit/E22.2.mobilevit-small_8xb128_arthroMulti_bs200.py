_base_ = [
    '../_base_/models/mobilevit/mobilevit_s_arthroMulti.py',
    '../_base_/datasets/arthroMulti_max500_imagenet_bs32.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/imagenet_bs256.py',
]

# no normalize for original implements
data_preprocessor = dict(
    # RGB format normalization parameters
    mean=[0, 0, 0],
    std=[255, 255, 255],
    # use bgr directly
    to_rgb=False,
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackMultiTaskInputs', multi_task_fields=('gt_label', )),
]

train_dataloader = dict(batch_size=200)

val_dataloader = dict(
    batch_size=180,
    dataset=dict(pipeline=test_pipeline),
)
test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=400)




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
        dict(type='WandbVisBackend',
            init_kwargs={
                'project': 'Mi1A2',
                'group': 'arthro_multi2',
                'name': 'E22.1.vit-base-p16_8xb128-coslr-100e_arthroMulti'
            })
    ]
)