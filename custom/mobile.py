import torch.nn as nn
import timm
from torch.nn import functional as F




mobile_models = {
    'mobilevit_s':640,
    'mobilevit_xs':640,
    'mobilevit_xxs':640,
    'mobilenetv2_035':640,
    'mobilenetv2_075':640,
    'mobilenetv2_100':640,
    'resnet5m':512,   
}


# mobile models declaration
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

    

