
import warnings
warnings.filterwarnings('ignore')

# torch and lightning imports
import torch
import pytorch_lightning as pl

from torchvision import transforms
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader

import pycimg
import multiprocessing
from typing import Tuple,Any

import random

def myloader(path):
    '''
    Lee CIMg y devuelve lista de PILS tantos como vistas tenga el fruto
    '''
    pils=pycimg.cimgread(path)
    return pils
    
# Esto es el data set    
class CImgFruitFolder(DatasetFolder):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, loader = myloader, extensions = ('.cimg',), **kwargs)
    
            
    def __getitem__(self, index: int) -> Tuple[Any, Any,Any]:
        path, target = self.samples[index]
        samples = self.loader(path) # Fruto: Lista de vistas
                
        samples_transformed = [] # cada vista sufre una aumentacion diferente
        for sample in samples:
            if self.transform is not None:
                sample = self.transform(sample)
                            
            samples_transformed.append(sample)
            
        sample = torch.stack(samples_transformed,axis=0)                
        if self.target_transform is not None:
            target = self.target_transform(target)

        # sample es un tensor con nviewsx3250x250
        # target es 0,1,2
        # path nombre del archivo
        return sample,target,path
 
def my_collate_fn(data): # Genera un batch a partir de una lista de frutos
    
    images = [d[0] for d in data]
    images = torch.concat(images, axis = 0) # tendra dimensiones numvistastotalbatch, 3,250,250
    
    nviews = [d[0].shape[0] for d in data] # contiene (nviews0, nviews1,,... ) con tantos elementos como frutos tenga el batch
    # Sirve para poder trocear luego por frutos
    
    labels = [d[1] for d in data]
    labels = torch.tensor(labels) #(5)

    paths = [d[2] for d in data]
    return { #(6)
        'images': images, 
        'label': labels,
        'nviews': nviews,
        'paths': paths
    }

def m_target_transform(target):
    return target


class FruitDataModule(pl.LightningDataModule):
    def __init__(self, train_set_folder = None, 
                 test_set_folder = None, 
                 predict_set_folder = None , 
                batch_size: int =5,  
                num_workers = -1,
                num_clases = None, **kwargs):
        super().__init__()

        print("Options in DataModule", kwargs)
        
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers > 0 else multiprocessing.cpu_count()-1
        
        transform_train = transforms.Compose([
        transforms.Resize((250,250)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(180) , 
                                  
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])      


        transform_test = transforms.Compose([
        transforms.Resize((250,250)),
              
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])        
        
        if train_set_folder is not None:
            self.train_dataset = CImgFruitFolder(train_set_folder,transform = transform_train )
        else:
            self.train_dataset= None
            
        if test_set_folder is not None:    
            self.val_dataset = CImgFruitFolder(test_set_folder,transform = transform_test )
        else:
            self.val_dataset = None
        
        if predict_set_folder is not None:     
            self.predict_dataset = CImgFruitFolder(predict_set_folder,transform = transform_test )
        else:
            self.predict_dataset = None

        if num_clases is None:
            self.num_classes = len(self.train_dataset.classes)
        else:
            self.num_classes=num_clases
            
        print(f"num clases = {self.num_classes}")
        if train_set_folder is not None:
            print(f"len total trainset =   {len(self.train_dataset )}")
            print(self.train_dataset.classes)
            print(self.train_dataset.class_to_idx)
        if test_set_folder is not None:
            print(f"len total testset =   {len(self.val_dataset )}")
        if predict_set_folder is not None:
            print(f"len total predictset =   {len(self.predict_dataset )}")            

        print("batch_size in FruitDataModule", self.batch_size)
        

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        return None
        
    def train_dataloader(self):
        print("batch_size in Dataloader train", self.batch_size)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=my_collate_fn)
    def predict_dataloader(self):
        print("batch_size in predict data loader", self.batch_size)
        return DataLoader(self.predict_dataset , batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=my_collate_fn)

    def val_dataloader(self):
        print("batch_size in Dataloader val", self.batch_size)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False,shuffle=False, collate_fn=my_collate_fn)



    @staticmethod
    def add_model_specific_args(parser):
        #parser = parent_parser.add_argument_group("model")
        #parser.add_argument("--data.train_set_csv", type=str, default='text_files/train_set_articulo.csv')
        #parser.add_argument("--data.test_set_csv", type=str, default='text_files/test_set_articulo.csv')
        return parser

    
