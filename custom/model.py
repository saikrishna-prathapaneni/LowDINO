import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.nn import functional as F
import collections



class Head(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 hidden_dim = 512,
                 bottleneck_dim = 256,
                 n_layers =3,
                 norm_last_layer=False,
                 init_weights=["normal",""] # yet to define
                 ) -> None:
        super().__init__()
        
        # create a Multilayer perceptron based on the layer number from in dim to out dim
       
        if n_layers ==1:
            self.mlp =nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            layers.append(nn.SELU())
            for _ in range(n_layers-2):
                layers.append(nn.Linear(hidden_dim,hidden_dim))
                layers.append(nn.SELU())
            layers.append(nn.Linear(hidden_dim,bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        
        
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim,out_dim,bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad=False
        
    def _init_weights(self,m):
        if isinstance(m,nn.Linear):
            nn.init.normal_(m.weight,std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias,0)

    def forward(self,x):
        x= self.mlp(x)
        x= F.normalize(x,dim=-1,p=2)
        x=self.last_layer(x)
        return x
    

class MultiCrop(nn.Module):
    """
    backbone: timm.models.vision_transformer.VisionTransformer
    new_head: head

    """

    def __init__(self,
                 backbone,
                 new_head,
                 mobile_head=False
                 ) -> None:
        super().__init__()
        self.mobile_head =mobile_head 

        #setting up the model
        self.backbone = backbone
        backbone.head= nn.Identity()
        self.new_head = new_head


    def forward(self,x):
        """
        x is List of torch.Tensor of shape (n_samples, 3,size,size)
        
        """
        n_crops = len(x)
        #print("len of batch ",len(x))
        concatenated_tensor = torch.cat(x,dim=0) 
        # (n_samples*n_crops, 3, size, size)
        # example batch size of 64 we have [640,3, 224,224] for size crops of 10: 2G,8L
        
        #print("shape of concat tensor",concatenated_tensor.shape)
        cls_embedding = self.backbone(concatenated_tensor) # (n_samples * n_crops, in_dim)
        #print(cls_embedding.shape, "cls embedding")
        logits =self.new_head(cls_embedding) # n_samples * n_crops, out_dim

        chunks = logits.chunk(n_crops) # n_crops * (n_samples,outdim)
        
        return chunks


class DinoLoss(nn.Module):
    def __init__(self,
                 out_dim, teacher_temp =0.04,student_temp=0.1, center_momentum =0.9
                 ) -> None:
        super(DinoLoss,self).__init__()
        self.student_temp = student_temp # teacher temperature should be scheduled
        self.teacher_temp = teacher_temp 
        self.center_momentum = center_momentum
        self.register_buffer("center",torch.zeros(1,out_dim))



    def forward(self, student_output, teacher_output):
        student_temp = [s / self.student_temp for s in student_output]
        teacher_temp = [(t - self.center) / self.teacher_temp for t in teacher_output]

        student_sm = [F.log_softmax(s, dim=-1) for s in student_temp]
        teacher_sm = [F.softmax(t, dim=-1).detach() for t in teacher_temp]

        total_loss = 0
        n_loss_terms = 0

        for t_ix, t in enumerate(teacher_sm):
            for s_ix, s in enumerate(student_sm):
                if t_ix == s_ix:
                    continue

                loss = torch.sum(-t * s, dim=-1)  # (n_samples,)
                total_loss += loss.mean()  # scalar
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center used for teacher output.
        Compute the exponential moving average.
        Parameters
        ----------
        teacher_output : tuple
            Tuple of tensors of shape `(n_samples, out_dim)` where each
            tensor represents a different crop.
        """
        batch_center = torch.cat(teacher_output).mean(
            dim=0, keepdim=True
        )  # (1, out_dim)
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )



def clip_gradients(model, clip=2.0):
    """Rescale norm of computed gradients.
    Parameters
    ----------
    model : nn.Module
        Module.
    clip : float
        Maximum norm.
    """
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)





    