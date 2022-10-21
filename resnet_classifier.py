import sys
import warnings
from pathlib import Path
from argparse import ArgumentParser
warnings.filterwarnings('ignore')

# torch and lightning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import SGD, Adam
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Tuple,Any

from dataLoad import FruitDataModule


# Here we define a new class to turn the ResNet model that we want to use as a feature extractor
# into a pytorch-lightning module so that we can take advantage of lightning's Trainer object.
# We aim to make it a little more general by allowing users to define the number of prediction classes.


    
class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes, resnet_version,
                optimizer='adam', lr=1e-3, batch_size=16,
                transfer=True, tune_fc_only=True):
        super().__init__()

        self.input_size=(256,256);
        self.mean_normalization = (0.5,)
        self.std_normalization = (0.5,)
        self.__dict__.update(locals())
        resnets = {
            18: models.resnet18, 34: models.resnet34,
            50: models.resnet50, 101: models.resnet101,
            152: models.resnet152
        }
        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers[optimizer]
        #instantiate loss criterion
        #self.criterion = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()
        # Using a pretrained ResNet backbone
        self.resnet_model = resnets[resnet_version](pretrained=transfer)
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features
        # replace final layer for fine tuning
        self.resnet_model.fc = nn.Linear(linear_size, num_classes) # num_classes solo defectos 

        if tune_fc_only: # option to only tune the fully-connected layers
            for child in list(self.resnet_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, X, nviews):
        Y_all_views = self.resnet_model(X)
        tmp = torch.split(Y_all_views, nviews)
        logits_fruit = torch.concat([torch.mean(fruit, axis = 0, keepdim= True) for fruit in tmp],axis = 0)
        return logits_fruit
    
    def criterion(self, logits, labels, weight_loss = 1.0):
        defect_logits = logits.mean(axis=1)
        labels_defect = (labels > 0).float()
        binaryLoss = nn.BCEWithLogitsLoss()
        loss = binaryLoss(defect_logits, labels_defect )
        
        defectuosos_logits = logits[labels > 0]
        defectuosos_labels = labels[labels > 0]
        
        if len(defectuosos_labels) > 0:
            tipodefectoLoss = nn.CrossEntropyLoss()
            loss += weight_loss * tipodefectoLoss(defectuosos_logits, defectuosos_labels)
        return loss
            
        

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)
    
    # def train_dataloader(self):
    #     # values here are specific to pneumonia dataset and should be changed for custom data
    #     transform = transforms.Compose([
    #             transforms.Resize(self.input_size),
    #             transforms.RandomHorizontalFlip(0.3),
    #             transforms.RandomVerticalFlip(0.3),
    #             transforms.RandomApply([   
    #                 transforms.RandomRotation(180)                    
    #             ]),
    #             transforms.ToTensor(),
    #             transforms.Normalize(self.mean_normalization, self.std_normalization)
    #     ])
    #     img_train = ImageFolder(self.train_path, transform=transform, loader=myloader)
    #     return DataLoader(img_train, batch_size=self.batch_size, shuffle=True)
    
        # def val_dataloader(self):
    #     # values here are specific to pneumonia dataset and should be changed for custom data
    #     transform = transforms.Compose([
    #             transforms.Resize(self.input_size),
    #             transforms.ToTensor(),
    #             transforms.Normalize(self.mean_normalization, self.std_normalization)
    #     ])
        
    #     img_val = ImageFolder(self.vld_path, transform=transform)
        
    #     return DataLoader(img_val, batch_size=1, shuffle=False)

    # def test_dataloader(self):
    #     # values here are specific to pneumonia dataset and should be changed for custom data
    #     transform = transforms.Compose([
    #             transforms.Resize((500,500)),
    #             transforms.ToTensor(),
    #             transforms.Normalize( self.mean_normalization, (0.23051,))
    #     ])
        
    #     img_test = ImageFolder(self.test_path, transform=transform)
        
    #     return DataLoader(img_test, batch_size=1, shuffle=False)    
    
    def training_step(self, batch, batch_idx):
        images = batch['images']
        labels = batch['labels']
        nviews = batch['nviews']
        logits = self(images, nviews)
        
        loss = self.criterion(logits, labels)
        prob_healthy = torch.sigmoid(logits.mean(axis=1))
        acc_healthy = ((prob_healthy > 0.5).long() == (labels > 0).long()).sum() / prob_healthy.shape[0]
        
        defectuosos_logits = logits[labels > 0]
        defectuosos_labels = labels[labels > 0]
        
        
        acc_tipo_defecto = (torch.argmax(defectuosos_logits,1) == defectuosos_labels) \
                .type(torch.FloatTensor).mean() if defectuosos_labels.shape[0] > 0 else 1.0
        # perform logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc_healthy", acc_healthy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc_tipo_defecto", acc_tipo_defecto, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss


    
    def validation_step(self, batch, batch_idx):
        images = batch['images']
        labels = batch['labels']
        nviews = batch['nviews']
        logits = self(images, nviews)
        
        loss = self.criterion(logits, labels)
        prob_healthy = torch.sigmoid(logits.mean(axis=1))
        acc_healthy = ((prob_healthy > 0.5).long() == (labels > 0).long()).sum() / prob_healthy.shape[0]
        
        defectuosos_logits = logits[labels > 0]
        defectuosos_labels = labels[labels > 0]
        
        
        acc_tipo_defecto = (torch.argmax(defectuosos_logits,1) == defectuosos_labels) \
                .type(torch.FloatTensor).mean() if defectuosos_labels.shape[0] > 0 else 1.0
        # perform logging
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc_healthy", acc_healthy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc_tipo_defecto", acc_tipo_defecto, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        


    
    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        if self.num_classes == 2:
            y = F.one_hot(y, num_classes=2).float()
        
        loss = self.criterion(preds, y)
        acc = (torch.argmax(y,1) == torch.argmax(preds,1)) \
                .type(torch.FloatTensor).mean()
        # perform logging
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_step=True, prog_bar=True, logger=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    # Required arguments
    parser.add_argument("--model", default = 18,
                        help="""Choose one of the predefined ResNet models provided by torchvision. e.g. 50""",
                        type=int)

    parser.add_argument("--num_epochs", default = 30, help="""Number of Epochs to Run.""", type=int)
    
    # Optional arguments
   
    parser.add_argument("-o", "--optimizer", help="""PyTorch optimizer to use. Defaults to adam.""", default='sgd')
    parser.add_argument("-lr", "--learning_rate", help="Adjust learning rate of optimizer.", type=float, default=1e-3)
    # parser.add_argument("-b", "--batch_size", help="""Manually determine batch size. Defaults to 16.""",
    #                     type=int, default=10)
    parser.add_argument("-tr", "--transfer",
                        help="""Determine whether to use pretrained model or train from scratch. Defaults to True.""",
                        action="store_true")
    parser.add_argument("-to", "--tune_fc_only", default=True, help="Tune only the final, fully connected layers.", action="store_true")
    parser.add_argument("-s", "--save_path", default='./out_models/', help="""Path to save model trained model checkpoint.""")
    parser.add_argument("-g", "--gpus", help="""Enables GPU acceleration.""", type=int, default=None)
    args = parser.parse_args()


    datamodule = FruitDataModule()

    # # Instantiate Model
    model = ResNetClassifier(num_classes = datamodule.num_classes, resnet_version = args.model,
                            optimizer = args.optimizer, lr = args.learning_rate,
                            #batch_size = args.batch_size,
                            transfer = args.transfer, tune_fc_only = args.tune_fc_only)
    # Instantiate lightning trainer and train model
    trainer_args = {'gpus': args.gpus, 'max_epochs': args.num_epochs}
    
    
    
    
    trainer = pl.Trainer(**trainer_args)
    
    trainer.fit(model, datamodule=datamodule)
    # Save trained model
    save_path = (args.save_path if args.save_path is not None else '/') + 'trained_model.ckpt'
    trainer.save_checkpoint(save_path)
