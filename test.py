import argparse
import json
import pathlib
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
import datetime

from custom.Augmentation import DataAugmentation


import matplotlib.pyplot as plt

if __name__=="__main__":
    
     
    img_train_small ="C:/Users/saipr/all projects/DL project/imagenette2-320/train"
    path_dataset_train = pathlib.Path(img_train_small)
    transform_aug = DataAugmentation(size=224, n_local_crops=10 - 2)
    dataset_train_aug = ImageFolder(path_dataset_train, transform=transform_aug)

    for i,imgs in enumerate(dataset_train_aug):
        for j in imgs:
            if type(j)==list:

                print(len(j))
                for k in j:
                    img_tensor = k
                    print(img_tensor.shape)
                    img=img_tensor.permute(1, 2, 0)
                    plt.imshow(img)
                    plt.show()
                print("sub img end")
            else:
                print("Int element",j)
