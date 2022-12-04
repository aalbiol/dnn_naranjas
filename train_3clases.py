
import warnings
from pathlib import Path
from argparse import ArgumentParser
warnings.filterwarnings('ignore')

# torch and lightning imports
import torch

import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger


from dataLoad import FruitDataModule

from pl_modelo_3clases import ResNetMILClassifier

# Here we define a new class to turn the ResNet model that we want to use as a feature extractor
# into a pytorch-lightning module so that we can take advantage of lightning's Trainer object.
# We aim to make it a little more general by allowing users to define the number of prediction classes.



if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--model", default = 18,
                        help="""Choose one of the predefined ResNet models provided by torchvision. e.g. 50""",
                        type=int)

    parser.add_argument("--num_epochs", default = 100, help="""Number of Epochs to Run.""", type=int)

    parser.add_argument("-o", "--optimizer", help="""PyTorch optimizer to use. Defaults to adam.""", default='sgd')
    parser.add_argument("-lr", "--learning_rate", help="Adjust learning rate of optimizer.", type=float, default=1e-3)
    parser.add_argument("-b", "--batch_size", help="""Manually determine batch size. Defaults to 16.""",
                         type=int, default=10)
    parser.add_argument("-tr", "--transfer",
                        help="""Determine whether to use pretrained model or train from scratch. Defaults to True.""",
                        action="store_true",
                        default=True)
    parser.add_argument("-to", "--tune_fc_only", default=False, help="Tune only the final, fully connected layers.", action="store_true")
    parser.add_argument("-s", "--save_path", default='./out_models/', help="""Path to save model trained model checkpoint.""")
    parser.add_argument("-i", "--initial_model", default=None, help="""Initial Model . If None Resnet is used""")
    parser.add_argument("-g", "--gpus", help="""Enables GPU acceleration.""", type=int, default= 0)
    parser.add_argument("--log_name",help="""WandB log name""", default='Resnet 18')
    parser.add_argument("--train_set_folder",help="""Folder containing the training data. Each subdirectory is a class""",
                        default='/home/aalbiol/reanotado_3_clases_con_Sara/train')

    parser.add_argument("--test_set_folder",help="""Folder containing the validation data. Each subdirectory is a class""",
                        default='/home/aalbiol/reanotado_3_clases_con_Sara/test')
    args = parser.parse_args()


    datamodule = FruitDataModule(batch_size=args.batch_size,train_set_folder = args.train_set_folder, 
    test_set_folder =  args.test_set_folder)

    # # Instantiate Model
    
    print("Tune Only Fully Connected:", args.tune_fc_only)
    
    model = ResNetMILClassifier(num_classes = datamodule.num_classes, resnet_version = args.model,
                            optimizer = args.optimizer, lr = args.learning_rate,
                            batch_size = args.batch_size,
                            transfer = args.transfer, tune_fc_only = args.tune_fc_only,
                            class_names = datamodule.train_dataset.classes,
                            save_intermediate="./intermediate_models/")
    # Continuar entrenamiento a partir de un punto
    if args.initial_model is not None:
        checkpoint = torch.load(args.initial_model)
        model.load_state_dict(checkpoint['state_dict'])



    # Instantiate lightning trainer and train model
    miwandb= WandbLogger(name=args.log_name, project='MILOranges')
    trainer_args = {'gpus': args.gpus, 'max_epochs': args.num_epochs, 'logger' : miwandb, 'auto_scale_batch_size':True}
    
    
    print('num_epochs:',args.num_epochs)
    
    trainer = pl.Trainer(**trainer_args)
    
    trainer.fit(model, datamodule=datamodule)
    # Save trained model
    save_path = (args.save_path if args.save_path is not None else '/') + 'Resnet_MIL_trained_model.ckpt'
    trainer.save_checkpoint(save_path)
