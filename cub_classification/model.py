import pytorch_lightning as pl
from torch import nn
import torch
import torch.nn.functional as F
from torch.optim import SGD

class CUBModel(pl.LightningModule):
    def __init__(
        self,
        num_classes = 200,
        train_classification = True,
        train_regression = True,
        classification_weight = 1.0,
        regression_weight = 1.0,
        lr = 1e-3
    ):

        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.train_classification = train_classification
        self.train_regression = train_regression
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        self.lr = lr

        # feature extractor
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # feature vector
        self.gap = nn.AdaptiveAvgPool2d(
            (56, 56)
        )

        # this predicts the model clas
        self.classifier = nn.Sequential(
            nn.Linear(8 * 56 * 56, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes)
        )

        # this predicts the bounding box
        self.regressor = nn.Sequential(
            nn.Linear(8 * 56 * 56, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )






    def forward(self, x):

        # feature extractor            
        x = self.conv1(x)
        x = F.relu(x) # non parametric
        x = self.pool(x) # non parametric

        x = self.conv2(x)
        x = F.relu(x) # non parametric
        x = self.pool(x) # non parametric

        x = self.gap(x)
        x = x.view(x.size(0), -1) # converting to 1-dimension before passing to classifier and bbox regressor

        # Classifier
        x_class = self.classifier(x)

        # Regressor
        x_reg = self.regressor(x)

        return x_class, x_reg

    def training_step(self, batch, batch_idx):

        images, (labels, bounding_boxes) = batch
        labels_pred, bounding_boxes_pred = self(images)

        loss = 0.0
        log_dict = {}

        if self.train_classification:
            classification_loss = F.cross_entropy(labels_pred, labels)
            loss += self.classification_weight * classification_loss
            log_dict["train_classification_loss"] = classification_loss

        if self.train_regression:
            regression_loss = F.mse_loss(bounding_boxes_pred, bounding_boxes)
            loss += self.regression_weight * regression_loss
            log_dict["train_regression_loss"] = regression_loss

        self.log_dict(log_dict, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):

        images, (labels, bounding_boxes) = batch
        labels_pred, bounding_boxes_pred = self(images)

        loss = 0.0
        log_dict = {}

        if self.train_classification:
            cls_loss = F.cross_entropy(labels_pred, labels)
            loss += self.classification_weight * cls_loss
            log_dict["val_classification_loss"] = cls_loss

        if self.train_regression:
            reg_loss = F.mse_loss(bounding_boxes_pred, bounding_boxes)
            loss += self.regression_weight * reg_loss
            log_dict["val_regression_loss"] = reg_loss

        if self.train_classification:
            preds = labels_pred.argmax(dim=1)
            acc = (preds == labels).float().mean()
            log_dict["val_accuracy"] = acc

        if self.train_regression:
            iou = self._batch_iou(bounding_boxes_pred, bounding_boxes).mean()
            log_dict["val_iou"] = iou

        if self.train_classification and self.train_regression:
            combined = (
                acc + iou
            ) / 2
        elif self.train_classification:
            combined = acc
        else:
            combined = iou

        log_dict["val_combined_metric"] = combined

        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.lr)
    
    @staticmethod
    def _batch_iou(box_a, box_b):
        """
        box_a, box_b : (B, 4) tensors  [x1, y1, x2, y2]
        returns       : (B,) IoU for each pair
        """
        # intersection
        x1 = torch.maximum(box_a[:, 0], box_b[:, 0])
        y1 = torch.maximum(box_a[:, 1], box_b[:, 1])
        x2 = torch.minimum(box_a[:, 2], box_b[:, 2])
        y2 = torch.minimum(box_a[:, 3], box_b[:, 3])

        inter_w = (x2 - x1).clamp(min=0)
        inter_h = (y2 - y1).clamp(min=0)
        inter = inter_w * inter_h

        # areas
        area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
        area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
        union = area_a + area_b - inter + 1e-7

        return inter / union
