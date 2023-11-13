import torch
import pytorch_lightning as pl
import os
import numpy as np
from PIL import Image
import math
from torchvision.models.detection import retinanet_resnet50_fpn

class Net(pl.LightningModule):

    def __init__(self, n_class = 4):
        super().__init__()

        self.model = retinanet_resnet50_fpn(pretrained = True)
        self.num_anchors = self.model.head.classification_head.num_anchors

        self.model.head.classification_head.num_classes = n_class

        self.cls_logits = torch.nn.Conv2d(256, self.num_anchors * n_class, kernel_size = 3, stride = 1, padding = 1)
        torch.nn.init.normal_(self.cls_logits.weight, std = 0.01)  # RetinaNetClassificationHead
        torch.nn.init.constant_(self.cls_logits.bias, - math.log((1 - 0.01) / 0.01))  # RetinaNetClassificationHead
        self.model.head.classification_head.cls_logits = self.cls_logits

        for p in self.model.parameters():
            p.requires_grad = False

        for p in self.model.head.classification_head.parameters():
            p.requires_grad = True

        for p in self.model.head.regression_head.parameters():
            p.requires_grad = True

        self.model.cuda()

    def forward(self, x, t = None):
        if self.training:
            return self.model(x, t)
        else:
            return self.model(x)


    def training_step(self, batch, batch_idx):
        x, t = batch
        losses = self(x, t)
        loss = sum(losses.values())
        self.log('train_loss', loss, on_step = True, on_epoch = True, prog_bar = True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, t = batch
        losses = self.train().forward(x, t)
        loss = sum(losses.values())
        self.log('val_loss', loss, on_step = False, on_epoch = True)


    def test_step(self, batch, batch_idx):
        x, t = batch
        losses = self.train().forward(x, t)
        loss = sum(losses.values())
        self.log('test_loss', loss, on_step = False, on_epoch = True)


    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params)
        return optimizer