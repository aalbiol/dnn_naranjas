
import warnings
warnings.filterwarnings('ignore')

# torch and lightning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import SGD, Adam

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from torchmetrics import ConfusionMatrix

import numpy as np



# Here we define a new class to turn the ResNet model that we want to use as a feature extractor
# into a pytorch-lightning module so that we can take advantage of lightning's Trainer object.
# We aim to make it a little more general by allowing users to define the number of prediction classes.



def m_accuracy(logits,labels,printout=False):
    preds_class = logits.argmax(axis = 1)
    acc=(preds_class==labels).type(torch.FloatTensor).mean()
    return acc
    
class ResNetMILClassifier(pl.LightningModule):
    def __init__(self, num_classes, resnet_version,
                optimizer='sgd', lr=1e-3, batch_size=16,
                transfer=True, tune_fc_only=True, class_names=None,
                save_intermediate=None):
        super().__init__()

        if class_names is not None:
            self.class_names=class_names
        else:
            self.class_names=list(range(num_classes))

        self.input_size=(256,256);
        self.save_intermediate=save_intermediate

        self.__dict__.update(locals())
        self.batch_size=batch_size
        resnets = {
            18: models.resnet18, 34: models.resnet34,
            50: models.resnet50, 101: models.resnet101,
            152: models.resnet152
        }
        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers[optimizer]
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
        self.confusion_matrix_train = ConfusionMatrix(num_classes=num_classes, normalize=None)
        self.confusion_matrix_val = ConfusionMatrix(num_classes=num_classes, normalize=None)
        self.class_weights=torch.ones(num_classes)
        self.class_weights[0]=6.0
        
        self.epoch_counter=1

    def forward(self, X, nviews):# Para training
        logits_all_views = self.resnet_model(X)
        probs_all_views = F.softmax(logits_all_views, dim = 1)
        
        # Agrupar lo que es de cada fruto
        logits_fruits = torch.split(logits_all_views, nviews)
        probs_fruits = torch.split(probs_all_views, nviews)

        logits_fruit=[] # Lista de los logits de la vista critica de cada fruto. Al final, tantos elementos como frutos
        for logits, probs in zip(logits_fruits, probs_fruits):
            # Sacar que vista es la mas critica: aquella donde las clases de defectos son maximas
            max_probs = torch.max(probs[:,1:],1)   
            vista_critica = torch.argmax(max_probs[0])

            logits_fruit.append(logits[(vista_critica,),:])
         
        logits_fruit = torch.concat(logits_fruit,axis = 0)
        
        return logits_fruit
    
    def criterion(self, logits, labels, weight_loss = 0.5):

        binaryLoss = nn.BCEWithLogitsLoss(reduction='mean')
        pesos=torch.FloatTensor(self.class_weights)
        #tipodefectoLoss = nn.CrossEntropyLoss(weight=pesos, reduction='mean')
        tipodefectoLoss = nn.CrossEntropyLoss(reduction='mean')

        loss = tipodefectoLoss(logits,labels)
        return loss
            
        

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)
    
    
    def training_step(self, batch, batch_idx):
        images = batch['images']
        labels = batch['label']
        nviews = batch['nviews']
        
        logits = self(images, nviews)

        loss = self.criterion(logits, labels)
        acc_train = m_accuracy(logits,labels)

        preds_class = logits.argmax(axis = 1)
       
        
        # perform logging
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc_train, on_step=False, on_epoch=True, prog_bar=True, logger=True)
 
        self.confusion_matrix_train.update(preds=preds_class, target=labels)
        return loss


    
    def validation_step(self, batch, batch_idx):
        images = batch['images']
        labels = batch['label']
        nviews = batch['nviews']
        logits = self(images, nviews)
        
        loss = self.criterion(logits, labels)
        preds_class = logits.argmax(axis = 1)
       
        acc_test = m_accuracy(logits,labels,True)
    
        # perform logging
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc_test, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.confusion_matrix_val.update(preds=preds_class, target=labels)

    
    def predict_step(self, batch, batch_idx, dataloader_idx=0): 
        ''' Con este metodo se predice la clase del fruto
        Se toma la vista con mas probabilidad de algun tipo de defecto
        Se mira la clase de dicha vista
        
        TODO: Se puede pensar en otras estrategias tale como contar el número de vistas clasificadas como defecto.
        O también consider la vista con el defecto mas serio. Por ejemplo si 1 vista es de 3ª y otra de 2ª y el resto buenas,
        se clasificaria como de 3ª
        '''
        images = batch['images']
        nviews = batch['nviews']
        paths = batch['paths'] 
        
        logits = self(images, nviews)
        preds_class = logits.argmax(axis = 1)
        return preds_class,paths    
    
    
    def on_validation_epoch_end(self, ) -> None:
        
        CM_val = self.confusion_matrix_val.compute()
        self.confusion_matrix_val.reset()
        
        CM_train = self.confusion_matrix_train.compute()
        self.confusion_matrix_train.reset()       
               
        CM_val = CM_val.cpu().numpy()
        CM_train = CM_train.cpu().numpy()
        print('Confusion Matrix Train:', CM_train)
        print('Confusion Matrix Val:', CM_val)
        
        self.epoch_counter += 1
        if self.epoch_counter %100 != 0:
            return
        if self.save_intermediate is not None:
            fname='model_batch_'+str(self.batch_counter)+'.ckpt'
            fullname=self.save_intermediate+fname
            torch.save(self.model.state_dict(), fullname)
            
