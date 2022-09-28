# model settings
keep_labels = ['fish', 'ihc']
label_dims = [2, 4]
model = dict(
    type='ImageClassifier',
    # pretrained='torchvision://resnet18',
    backbone=dict(
        type='ResNet',
        in_channels=6,
        depth=18,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=-1,
        style='pytorch'),
    neck=None,
    bbox_head=dict(
            type='SimpleImageClassificationMultiHead',
            num_classes_lst=[2, 4],
            head_names=['fish', 'ihc'],
            dropout_probs=[0.8, 0.8],
            loss_classification=[dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                                 dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)],
            input_size=128,
            poolsize=4,
            cls_feat_dim=512
        ),
    )
# training and testing settings
train_cfg = dict()
test_cfg = dict()
# dataset settings
dataset_type = 'HER2Classification'
data_root = 'HER2Classification/'
train_pipeline = [
    dict(type='LoadHer2FromFile', keep_labels=keep_labels, label_dims=label_dims, with_image=True, generate_bkg_img=0.2),
    dict(type='RandomRotate', min_angle=-180, max_angle=180, keep_rotate_angle=False),
    dict(type='RandomCrop', crop_size=(512, 512)),
    dict(type='RandomFlip', flip_ratio=0.3, direction=["horizontal", "vertical"]),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_labels'],
         meta_keys=['filename']),
]
test_pipeline = [
    dict(type='LoadHer2FromFile', keep_labels=keep_labels, label_dims=label_dims, with_image=True),
    dict(type='CenterCropWithPad', crop_size=(512, 512)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_labels'],
         meta_keys=['filename'])
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'data_20220627.csv',
        img_prefix=data_root,
        fold=fold,
        keep_labels=keep_labels,
        label_dims=label_dims,
        uniform_sampling=False,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'data_20220627.csv',
        keep_labels=keep_labels,
        label_dims=label_dims,
        uniform_sampling=False,
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'data_20220627.csv',
        keep_labels=keep_labels,
        label_dims=label_dims,
        uniform_sampling=False,
        img_prefix=data_root,
        pipeline=test_pipeline))
# optimizer
# optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.000001)
optimizer = dict(type='Adam', lr=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
        policy='CosineAnnealing',
        warmup='linear',
        warmup_iters=64,
        warmup_ratio=0.1,
        min_lr_ratio=1e-8)

checkpoint_config = dict(interval=40)

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook')
    ])
# runtime settings
total_epochs = 200
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/her2classification'
load_from = f'./work_dirs/her2classification/latest.pth'
resume_from = None
workflow = [('train', 1), ('val', 1)]
evaluation = dict(interval=1, metric='her2_accuracy')
