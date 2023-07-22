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