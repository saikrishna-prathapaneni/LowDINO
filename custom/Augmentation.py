import random
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image,ImageFilter,ImageOps

# Data Augmentation techniques have been taken from original DINO implementation

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
        


class DataAugmentation(object):
    def __init__(self,
                 global_crop_scale = (0.4,1.),
                 local_crop_scale = (0.05,0.4),
                 n_local_crops =8,
                 size =224
                 ) -> None:
        self.n_local_crops = n_local_crops
        self.global_crop_scale = global_crop_scale
        self.local_crop_scale = local_crop_scale

        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        print(global_crop_scale)
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=self.global_crop_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.GaussianBlur(kernel_size = 5, sigma =(0.1,2)),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(size =size, scale=self.global_crop_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        # size should be 96 according to original paper
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=self.local_crop_scale, interpolation=Image.BICUBIC), 
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.n_local_crops):
            crops.append(self.local_transfo(image))
        return crops



#test image
# if __name__=="__main__":
#    # Load an example image
#     image = Image.open("example.jpg")

#     # Initialize the DataAugmentation class
#     data_augmentation = DataAugmentation()

#     # Get the crops
#     crops = data_augmentation(image)

#     # Visualize the crops
#     fig, axes = plt.subplots(3, 3, figsize=(10, 10))
#     for i, ax in enumerate(axes.flat):
#         ax.imshow(crops[i].reshape(224,224,3))
#         ax.axis("off")
#     plt.show()