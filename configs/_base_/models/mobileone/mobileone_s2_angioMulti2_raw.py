model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MobileOne',
        arch='s2',
        out_indices=(3, ),
        # init_cfg=dict(type='Pretrained', checkpoint="/modelZoo/mobileone-s2_8xb32_in1k_20221110-9c7ecb97.pth", prefix='backbone',)
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(  # Multi-task head
        type='MultiTaskHead',
        task_heads={
            'L10': dict(type='LinearClsHead', num_classes=5906, in_channels=2048, 
                        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'), 
                        init_cfg=[dict(type='TruncNormal', layer='Linear', std=2e-5)]),
            'L20': dict(type='LinearClsHead', num_classes=1922, in_channels=2048, 
                        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'), 
                        init_cfg=[dict(type='TruncNormal', layer='Linear', std=2e-5)]),
            'L30': dict(type='LinearClsHead', num_classes=221, in_channels=2048, 
                        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
                        init_cfg=[dict(type='TruncNormal', layer='Linear', std=2e-5)]),
            'L40': dict(type='LinearClsHead', num_classes=50, in_channels=2048, 
                        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
                        init_cfg=[dict(type='TruncNormal', layer='Linear', std=2e-5)]),
        },
    ))
