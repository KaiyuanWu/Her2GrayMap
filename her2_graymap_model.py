from torch import nn
import torch
import numpy as np
from mmdet.models import BaseDetector
from mmdet.models import DETECTORS, HEADS, build_backbone, build_head, build_neck, build_loss
from mmcv.cnn import normal_init, kaiming_init, constant_init


@DETECTORS.register_module()
class Her2Classifier(BaseDetector):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Her2Classifier, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(ImageClassifier, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        return [self.bbox_head.aug_test(feats, img_metas, rescale=rescale)]

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_test(self, img, *args, img_metas=None, **kwargs):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        logits = self.bbox_head(x)
        if isinstance(logits, (list, )):
            # 多任务分类模型
            probs = [torch.softmax(l, dim=-1) for l in logits]
        else:
            probs = torch.softmax(logits, dim=-1)
        return probs

    def forward_train(self,
                      img,
                      img_metas=None,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        super(Her2Classifier, self).forward_train(img, img_metas)
        if gt_masks is not None:
            gt_masks = np.concatenate([np.stack(sample, axis=0) for sample in gt_masks])
            gt_masks = np.rollaxis(gt_masks, -1, 1)
            gt_masks = torch.tensor(gt_masks).to(img.device)
            img = gt_masks
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses


@HEADS.register_module()
class Her2ClassificationMultiHead(nn.Module):
    def __init__(self,
                 num_classes_lst,
                 head_names,
                 loss_classification,
                 dropout_probs=None, input_size=256, poolsize=32, poolstride=2,
                 cls_feat_dim=512, train_cfg=None, test_cfg=None):
        super(Her2ClassificationMultiHead, self).__init__()
        self._num_classes_lst = num_classes_lst
        self._head_names = head_names
        self._poolsize = poolsize
        self._poolstride = poolstride
        self._num_heads = len(num_classes_lst)
        self._input_size = input_size
        self._cls_feat_dim = cls_feat_dim
        self._dropout_probs = dropout_probs
        self._loss_classification = nn.ModuleList([build_loss(loss_config) for loss_config in loss_classification])
        self._init_layers()
        self._train_cfg = train_cfg
        self._test_cfg = test_cfg

    def _init_layers(self):
        self._classification_head = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.PReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.PReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.PReLU(),
                nn.MaxPool2d(kernel_size=self._poolsize, stride=self._poolstride, padding=0),
                nn.Flatten(start_dim=1),
                nn.Linear(in_features=self._cls_feat_dim, out_features=self._input_size),
                nn.PReLU(),
                nn.Dropout(p=self._dropout_probs[i]),
                nn.Linear(in_features=self._input_size, out_features=self._num_classes_lst[i]),
            ) for i in range(self._num_heads)])

    def init_weights(self):
        for m in self._classification_head:
            for n in m:
                if isinstance(n, (nn.Linear,)):
                    normal_init(n, mean=0, std=0.01, bias=0)

    def forward(self, x, img_metas=None,
                gt_bboxes=None,
                gt_labels=None,
                gt_bboxes_ignore=None):
        # neck的第1个输出
        x = x[0]
        x = [head(x) for head in self._classification_head]
        return x

    def forward_train(self,
                      x,
                      img_metas=None,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None):
        gt_labels = torch.cat(gt_labels)
        gt_labels = torch.split(gt_labels, self._num_classes_lst, dim=1)
        feats = self.forward(x)
        loss = {"loss_"+loss_name: self._loss_classification[i](feats[i], gt_labels[i])
                for i, loss_name in enumerate(self._head_names)}
        return loss
