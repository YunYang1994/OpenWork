

lr_mult   = 1
scheduler = [
    dict(type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        end=1500),
    dict(type='MultiStepLR',
        by_epoch=True,
        milestones=[30*lr_mult, 45*lr_mult],
        gamma=0.1)
    ]

train_cfg = dict(
    max_epochs=50*lr_mult,
    val_interval=10*lr_mult,
    by_epoch=True
    )

default_hooks = dict(
        logger=dict(type='LoggerHook', interval=50),
        checkpoint=dict(type='CheckpointHook', interval=2),
    )

work_dir  = "./work_dir"
norm_cfg  = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
optimizer = dict(optimizer=dict(type='SGD', lr=0.001, momentum=0.9))

dataset = dict(
    batch_size=16,
    train=dict(
        type='CifarDataset',
        root_dir='/cephFS/yangyun/data/cifar10',
        norm_cfg=norm_cfg,
        training=True,
    ),
    valid=dict(
        type='CifarDataset',
        root_dir='/cephFS/yangyun/data/cifar10',
        norm_cfg=norm_cfg,
        training=False,
    )
)

model = dict(type='ResNet18')

