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
import wandb.plot
from pytorch_lightning.loggers import WandbLogger
from typing import Tuple,Any

from dataLoad import FruitDataModule
from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt
import numpy as np



# Here we define a new class to turn the ResNet model that we want to use as a feature extractor
# into a pytorch-lightning module so that we can take advantage of lightning's Trainer object.
# We aim to make it a little more general by allowing users to define the number of prediction classes.



def m_accuracy(logits,labels,printout=False):

    preds_class = logits.argmax(axis = 1)


    acc=(preds_class==labels).type(torch.FloatTensor).mean()
    return acc
    
class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes, resnet_version,
                optimizer='adam', lr=1e-3, batch_size=16,
                 class_names=None):
        super().__init__()

        if class_names is not None:
            self.class_names=class_names
        else:
            self.class_names=list(range(num_classes))

        self.input_size=(256,256);
        #self.mean_normalization = (0.5,)
        #self.std_normalization = (0.5,)
        self.__dict__.update(locals())
        self.batch_size=batch_size
        resnets = {
            18: models.resnet18, 34: models.resnet34,
            50: models.resnet50, 101: models.resnet101,
            152: models.resnet152
        }

        self.resnet_model = resnets[resnet_version](pretrained=False)
        
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features
        # replace final layer for fine tuning
        self.resnet_model.fc = nn.Linear(linear_size, num_classes) # num_classes solo defectos 


    def forward(self, X, nviews): # X:batch con varios frutos nviews
        logits_all_views = self.resnet_model(X)
        probs_all_views = F.softmax(logits_all_views, dim = 1)
        logits_fruits = torch.split(logits_all_views, nviews)
        probs_fruits = torch.split(probs_all_views, nviews)
        #tmp = torch.split(Y_all_views, nviews)
        
        #print("logits_views:", Y_all_views)
        
        # Aqui es donde se fusionan las instancias (vistas) para obtener un logit por fruto y categoría
        # Se pueden fusionar por max o mean . La fusion debe ser conmutativa
        #logits_fruit = torch.concat([torch.mean(fruit, axis = 0, keepdim= True) for fruit in tmp],axis = 0)
        logits_fruit=[]
        for logits, probs in zip(logits_fruits, probs_fruits):
            max_probs = torch.max(probs[:,1:],1)
            fila = torch.argmax(max_probs[0])

        
            logits_fruit.append(logits[(fila,),:])
         # El primer elemento corresponde a la clase bueno y deberá ignorarse en criterion   
        logits_fruit = torch.concat(logits_fruit,axis = 0)
        
        return logits_fruit
    
    def criterion(self, logits, labels, weight_loss = 0.5):
        pass
            
    
    
    def training_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images = batch['images']
        
        nviews = batch['nviews']
        logits = self(images, nviews)
        paths = batch['paths']      
        preds_class = logits.argmax(axis = 1)
        return preds_class,paths
    
    def validation_step(self, batch, batch_idx):
        images = batch['images']
        labels = batch['label']
        nviews = batch['nviews']
        logits = self(images, nviews)
        
        loss = self.criterion(logits, labels)
        preds_class = logits.argmax(axis = 1)
       
        acc_test = m_accuracy(logits,labels,True)
        #acc_test_good_bad= ((predictions>0) == (labels>0)).type(torch.FloatTensor).mean() 
        # perform logging
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc_test, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #self.log("val_acc_good_bad", acc_test_good_bad, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.confusion_matrix_val.update(preds=preds_class, target=labels)

        
    def on_validation_epoch_end(self, ) -> None:
        pass
        


    
    def test_step(self, batch, batch_idx):
        pass


if __name__ == "__main__":
    parser = ArgumentParser()
    # Required arguments
    parser.add_argument("--model", default = 18,
                        help="""Choose one of the predefined ResNet models provided by torchvision. e.g. 50""",
                        type=int)

    parser.add_argument("--num_epochs", default = 3, help="""Number of Epochs to Run.""", type=int)
    parser.add_argument("-b", "--batch_size", help="""Manually determine batch size. Defaults to 16.""",
                         type=int, default=10)
    # Optional arguments
    parser.add_argument("-i", "--initial_model", default=None, help="""Initial Model . If None Resnet is used""")
    parser.add_argument("-g", "--gpus", help="""Enables GPU acceleration.""", type=int, default=None)

    parser.add_argument("-d", "--directory", default=None, help="""Name of folder with images for prediction""")
    

    args = parser.parse_args()

    predict_set_folder='/home/aalbiol/reanotado_3_clases_con_Sara/test'

    if args.directory is not None:
        predict_set_folder = args.directory
        print('Directory:', predict_set_folder)

    datamodule = FruitDataModule(batch_size=args.batch_size,
    train_set_folder = '/home/aalbiol/reanotado_3_clases_con_Sara/train', 
    test_set_folder = '/home/aalbiol/reanotado_3_clases_con_Sara/test',
    predict_set_folder=predict_set_folder)


    model = ResNetClassifier(num_classes = datamodule.num_classes, resnet_version = args.model, 
                            batch_size = args.batch_size,
                             class_names = datamodule.train_dataset.classes)
    if args.initial_model is not None:
        checkpoint = torch.load(args.initial_model)
        model.load_state_dict(checkpoint['state_dict'])

    trainer_args = {'gpus': args.gpus}
    trainer = pl.Trainer(**trainer_args)
    predictions = trainer.predict(model, datamodule.predict_dataloader())
    class_names = datamodule.train_dataset.classes

    name='predictions.txt'
    f = open(name, "w")

    for p in predictions: # Cada batch
        fichs = p[1]
        preds = p[0]
        for k in range(len(fichs)):
            caso=(fichs[k],class_names[preds[k]])
            print(caso)
            f.write(caso[0] +','+caso[1]+'\n')
    
    f.close()
    


