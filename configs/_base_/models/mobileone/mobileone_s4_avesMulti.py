model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MobileOne',
        arch='s4',
        out_indices=(3, ),
        init_cfg=dict(type='Pretrained', checkpoint="/modelZoo/mobileone-s4_8xb32_in1k_20221110-28d888cb.pth")
    ),
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
    ),)
