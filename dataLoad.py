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
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Tuple,Any
import pycimg
import multiprocessing

def myloader(path):
    '''
    Lee CIMg y devuelve lista de PILS
    '''
    pils=pycimg.cimgread(path)
    return pils
    
# Esto es el data set    
class CImgFruitFolder(DatasetFolder):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, loader = myloader, extensions = ('.cimg',), **kwargs)
        
     

        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        samples = self.loader(path) #Lista de imagenes
        samples_transformed = []
        for sample in samples:
            if self.transform is not None:
                sample = self.transform(sample)
                            
            samples_transformed.append(sample)
            
        sample = torch.stack(samples_transformed,axis=0)                
        if self.target_transform is not None:
            target = self.target_transform(target)


        # Tensor gordo
        return sample,target
 
def my_collate_fn(data):
    
    images = [d[0] for d in data]
    images = torch.concat(images, axis = 0)
    
    nviews = [d[0].shape[0] for d in data]
    
    labels = [d[1] for d in data]
    labels = torch.tensor(labels) #(5)
    return { #(6)
        'images': images, 
        'label': labels,
        'nviews': nviews
    }
    
class FruitDataModule(pl.LightningDataModule):
    def __init__(self, train_set_folder = 'orange_data/train' , 
                test_set_folder = 'orange_data/test',
                batch_size: int =5,  
                imsize = (256,256), 
                num_workers = -1, **kwargs):
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
        
     

        self.train_dataset  = CImgFruitFolder(train_set_folder,transform = transform_train)
        
        self.val_dataset  =  CImgFruitFolder(test_set_folder,transform = transform_test )

        self.num_classes = len(self.train_dataset.classes)
    
        print(f"num clases = {self.num_classes}")
        print(f"len total trainset =   {len(self.train_dataset )}")
        print(f"len total testset =   {len(self.val_dataset )}")
        print(self.train_dataset.classes)
        print(self.train_dataset.class_to_idx)
        print("batch_size in FruitDataModule", self.batch_size)
        

        self.save_hyperparameters()

    def prepare_data(self):
        # if not pathlib.Path(self.root_images).exists():
        #     print("Get fisabio covid19 dataset")
        pass

    def setup(self, stage=None):
        # build dataset
        # caltect_dataset = ImageFolder('Caltech101')
        # # split dataset
        # self.train, self.val, self.test = random_split(caltect_dataset, [6500, 1000, 1645])
        # self.train.dataset.transform = self.augmentation
        # self.val.dataset.transform = self.transform
        # self.test.dataset.transform = self.transform
        #print("Nothing to do in setup datasets, partitions already given")
        return None
        
    def train_dataloader(self):
        print("batch_size in Dataloader train", self.batch_size)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=my_collate_fn)

    def val_dataloader(self):
        print("batch_size in Dataloader train", self.batch_size)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False,shuffle=False, collate_fn=my_collate_fn)

    # def test_dataloader(self):
    #     return DataLoader(self.val_dataset, batch_size=self.batch_size,  num_workers=self.num_workers, drop_last=False,shuffle=False)

    @staticmethod
    def add_model_specific_args(parser):
        #parser = parent_parser.add_argument_group("model")
        #parser.add_argument("--data.train_set_csv", type=str, default='text_files/train_set_articulo.csv')
        #parser.add_argument("--data.test_set_csv", type=str, default='text_files/test_set_articulo.csv')
        return parser

    
