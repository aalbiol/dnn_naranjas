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

import pytorch_lightning as pl
import wandb
import wandb.plot
from pytorch_lightning.loggers import WandbLogger

from pl_modelo_3clases import ResNetMILClassifier
from dataLoad import FruitDataModule




# Here we define a new class to turn the ResNet model that we want to use as a feature extractor
# into a pytorch-lightning module so that we can take advantage of lightning's Trainer object.
# We aim to make it a little more general by allowing users to define the number of prediction classes.


if __name__ == "__main__":
    parser = ArgumentParser()
    # Required arguments
    parser.add_argument("--resnetmodel", default = 18,
                        help="""Choose one of the predefined ResNet models provided by torchvision. e.g. 50""",
                        type=int)

    
    parser.add_argument("-b", "--batch_size", help="""Manually determine batch size. Defaults to 16.""",
                         type=int, default=10)
    # Optional arguments
    parser.add_argument("-m", "--model", required=True,help="""Model""")
    parser.add_argument("-g", "--gpus", help="""Enables GPU acceleration.""", type=int, default=None)

    parser.add_argument("-d", "--directory", default=None, help="""Name of folder with images for prediction""")
    parser.add_argument("-n", "--num_classes", default=3, help="""Number of classes""")
    
    args = parser.parse_args()

    predict_set_folder='/home/aalbiol/reanotado_3_clases_con_Sara/test'


    print('********* Directory to predict: ', predict_set_folder)

    datamodule = FruitDataModule(batch_size=args.batch_size,
    train_set_folder=None,
    test_set_folder=None,                             
    predict_set_folder=args.directory,
    num_clases=3)

    class_names = ['1a','2a','3a']
    model = ResNetMILClassifier(num_classes = args.num_classes, resnet_version = args.resnetmodel, 
                            batch_size = args.batch_size,
                            class_names = class_names)
    
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['state_dict'])

    trainer_args = {'gpus': args.gpus}
    trainer = pl.Trainer(**trainer_args)
    
    predictions = trainer.predict(model, datamodule.predict_dataloader())
    

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
    


