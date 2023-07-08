import torchvision.models as models
import torch
import os
import wandb
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

from pytorch_metric_learning import losses, miners, samplers

BATCH_SIZE = 1024
VALID_PCT = 0.15
DATASET = '/home/frmich23/work/datasets/v5'
EMBEDDING_SIZE = 128
M_PER_CLASS = 32

from torchvision.datasets import ImageFolder
from torchvision import transforms
import multiprocessing as mp

full_dataset = ImageFolder(root=DATASET, transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float)
]))

train_size = int((1 - VALID_PCT) * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=mp.cpu_count(),
    shuffle=True,
    # sampler=samplers.MPerClassSampler(train_dataset.dataset.classes, M_PER_CLASS),
)

val_data_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=mp.cpu_count(),
    shuffle=False,
    # sampler=samplers.MPerClassSampler(val_dataset.dataset.classes, M_PER_CLASS),
)


class WriterINet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        model = models.resnet18(pretrained=False)
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        model.fc = torch.nn.Linear(in_features=512, out_features=EMBEDDING_SIZE, bias=True)
        self.model = model
        self.loss_func = losses.CircleLoss()

    def forward(self, data=None, get_encodings=False):
        x = self.model(data)
        return x

    def training_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.forward(data)
        loss = self.loss_func(outputs, targets)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.forward(data)
        test_loss = self.loss_func(outputs, targets)
        self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.forward(data)
        loss = self.loss_func(outputs, targets)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



wandb.login(key="")
# wandb.finish()
wandb_logger = WandbLogger(project="Handwrite embedding", log_model="all")

model = WriterINet()
#model = WriterINet.load_from_checkpoint(checkpoint_path="")

trainer = pl.Trainer(logger=wandb_logger)
trainer.fit(model, train_data_loader, val_data_loader)
# trainer.save_checkpoint("30_06_2023-resnet18-lighting_2.ckpt")