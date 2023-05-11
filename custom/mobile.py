import torch
import torch.nn as nn
import timm
from torch.nn import functional as F
# from model import Head, MultiCrop,DinoLoss
# from Augmentation import DataAugmentation
# from PIL import ImagePath
# from torchvision.datasets import ImageFolder
# import pathlib
# from torch.utils.data import DataLoader, SubsetRandomSampler
# model and dim values



mobile_models = {
    'mobilevit_s':640,
    'mobilevit_xs':640,
    'mobilevit_xxs':640,
    'mobilenetv2_035':640,
    'mobilenetv2_075':640,
    'mobilenetv2_100':640,
    'resnet5m':512,   
}

class mobilenet(nn.Module):
    def __init__(self,
                 model:str = 'mobilevit_s',
                 pretrained=False):
        super(mobilenet,self).__init__()
        self.backbone = timm.create_model(model,pretrained=pretrained)
        self.backbone.reset_classifier(0)
        self.num_features = self.backbone.num_features

    def forward(self,x):
        x = self.backbone(x)
        return x



# if __name__=="__main__":
#     m=torch.randn(1,3,224,224)
#     model_key = list(mobile_models.keys())[0]
#     model = mobilenet('mobilevit_s',pretrained=False)

 
#     path_dataset_train = pathlib.Path("data/imagenette2-320/train")
#     dataset_train_aug = ImageFolder(path_dataset_train, transform=DataAugmentation(size=224, n_local_crops=8))


#     print(len(dataset_train_aug))

#     data_loader_train_aug = DataLoader(
#         dataset_train_aug,
#         batch_size=1,
#         shuffle=True,
#         drop_last=True,
#         num_workers=2,
#         pin_memory=True,
#     )
#     model1 = MultiCrop(model,Head(640,1024))
#     model2 = MultiCrop(model,Head(640,1024))
#     for p in model1.parameters():
#         p.requires_grad = False
#     model1.to('cuda')
#     model2.to('cuda')

#     loss_inst =DinoLoss(1024)
#     for i, (batch,_) in enumerate(data_loader_train_aug):
#         print(len(batch), len(_), i)
#         images = [img.to('cuda') for img in batch]

#         s = model1.forward(images)
#         t = model2.forward(images[:2])
#         print("shape of output logits s:",len(s))
#         print("shape of output logits t:",len(t))
#         loss = loss_inst(s, t).to('cuda')
#         print(loss)
  
#     test = torch.randn(1,3,224,224)
    

    

#     data = model.forward(list(test))
#     print(data.shape)
