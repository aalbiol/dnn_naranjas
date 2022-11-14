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
import wandb
from pytorch_lightning.loggers import WandbLogger
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
        self.batch_size=batch_size
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
        m_softmax = nn.Softmax(dim=1)
        softmax_prob = m_softmax (Y_all_views)
        tmp = torch.split(softmax_prob, nviews)
        
        #print("logits_views:", Y_all_views)
        
        # Aqui es donde se fusionan las instancias (vistas) para obtener un logit por fruto y categoría
        # Se pueden fusionar por max o mean . La fusion debe ser conmutativa
        #logits_fruit = torch.concat([torch.mean(fruit, axis = 0, keepdim= True) for fruit in tmp],axis = 0)
        logits_fruit=[]
        for fruit in tmp:
            valsmax,posmax=torch.max(fruit,0,keepdim=True)
            logits_fruit = valsmax
         # El primer elemento corresponde a la clase bueno y deberá ignorarse en criterion   
        logits_fruit = torch.concat(logits_fruit,axis = 0)
        
        return logits_fruit
    
    def criterion(self, logits, labels, weight_loss = 0.5):
        #axis 0: batch_element
        #axis 1: categorias.El primer elemento el de la categoria buena
        #print("criterion_logits:",logits.shape)
        #print("criterion_labels:",labels.shape)

        
        target = logits * 0

        for count, value in enumerate(labels):
            target[count,value] = 1

        logits = logits[:,1:] # Quitar la categoría normal
        target = target[:,1:]    

        binaryLoss = nn.BCEWithLogitsLoss(reduction='mean')
        tipodefectoLoss = nn.CrossEntropyLoss(reduction='mean')

        loss = binaryLoss(logits,target)
        #loss = tipodefectoLoss(logits,labels)
        return loss
            
        

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)
    

    
    def training_step(self, batch, batch_idx):
        images = batch['images']
        labels = batch['label']
        nviews = batch['nviews']
        
 
        logits = self(images, nviews)

        #print('batch labels: ',labels)
        #print('batch nviews: ',nviews)       
        #print('batch logits: ',logits)

        loss = self.criterion(logits, labels)
 
        

        
        predictions=torch.argmax(logits,1) 
        acc_train = (predictions== labels).type(torch.FloatTensor).mean() 
        #acc_train_good_bad= ((predictions>0) == (labels>0)).type(torch.FloatTensor).mean() 
        # perform logging
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc_train, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #self.log("train_acc_good_bad", acc_train_good_bad, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #wandb.log({'accuracy': train_acc, 'loss': loss})
        #self.log("train_acc_healthy", acc_healthy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log("train_acc_tipo_defecto", acc_tipo_defecto, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss


    
    def validation_step(self, batch, batch_idx):
        images = batch['images']
        labels = batch['label']
        nviews = batch['nviews']
        logits = self(images, nviews)
        
        loss = self.criterion(logits, labels)
        predictions=torch.argmax(logits,1)
        acc_test = (predictions == labels).type(torch.FloatTensor).mean() 
        #acc_test_good_bad= ((predictions>0) == (labels>0)).type(torch.FloatTensor).mean() 
        # perform logging
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc_test, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #self.log("val_acc_good_bad", acc_test_good_bad, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        


    
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

    parser.add_argument("--num_epochs", default = 3, help="""Number of Epochs to Run.""", type=int)
    
    # Optional arguments
   
    parser.add_argument("-o", "--optimizer", help="""PyTorch optimizer to use. Defaults to adam.""", default='sgd')
    parser.add_argument("-lr", "--learning_rate", help="Adjust learning rate of optimizer.", type=float, default=1e-3)
    parser.add_argument("-b", "--batch_size", help="""Manually determine batch size. Defaults to 16.""",
                         type=int, default=2)
    parser.add_argument("-tr", "--transfer",
                        help="""Determine whether to use pretrained model or train from scratch. Defaults to True.""",
                        action="store_true")
    parser.add_argument("-to", "--tune_fc_only", default=True, help="Tune only the final, fully connected layers.", action="store_true")
    parser.add_argument("-s", "--save_path", default='./out_models/', help="""Path to save model trained model checkpoint.""")
    parser.add_argument("-g", "--gpus", help="""Enables GPU acceleration.""", type=int, default=None)
    parser.add_argument("--log_name",help="""WandB log name""", default='REsnet 18')

    args = parser.parse_args()


    datamodule = FruitDataModule(batch_size=args.batch_size)

    # # Instantiate Model
    model = ResNetClassifier(num_classes = datamodule.num_classes, resnet_version = args.model,
                            optimizer = args.optimizer, lr = args.learning_rate,
                            batch_size = args.batch_size,
                            transfer = args.transfer, tune_fc_only = args.tune_fc_only)
    # Instantiate lightning trainer and train model
    miwandb= WandbLogger(name=args.log_name, project='MILOranges')
    trainer_args = {'gpus': args.gpus, 'max_epochs': args.num_epochs, 'logger' : miwandb}
    
    
    
    print('num_epochs:',args.num_epochs)
    
    trainer = pl.Trainer(**trainer_args)
    
    trainer.fit(model, datamodule=datamodule)
    # Save trained model
    save_path = (args.save_path if args.save_path is not None else '/') + 'trained_model.ckpt'
    trainer.save_checkpoint(save_path)
