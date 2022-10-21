from dataLoad import  FruitDataModule
import torch



if __name__=="__main__":
    # cread data moudule
    print("Hola")
    datamodule = FruitDataModule()
    
    train_dataloader = datamodule.train_dataloader()
    
    
    for sample in iter(train_dataloader):
        
        print(sample['images'].shape)
        print(sample['nviews'])
        print(sample['label'])
        