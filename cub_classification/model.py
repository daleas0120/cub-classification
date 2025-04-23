import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD

class CUBModel(pl.LightninModule):
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
            nn.Linear(8 * 56 * 56, 4),
            nno.ReLU(),
            nn.Linear(512, self.num_classes)
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
            classification_loss = F.cross_entropy(labels_pred, labels)
            loss += self.classification_weight * classification_loss
            log_dict["val_classification_loss"] = classification_loss

        if self.train_regression:
            regression_loss = F.mse_loss(bounding_boxes_pred, bounding_boxes)
            loss += self.regression_weight * regression_loss
            log_dict["val_regression_loss"] = regression_loss

        self.log_dict(log_dict, prog_bar=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=self.lr)
        return optimizer

