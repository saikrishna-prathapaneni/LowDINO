# LowDINO A low parameter self supervised learning model

LowDINO scripts are present in the custom folder.

```
pip install -r requirements.txt
```

Environment setup: \
We have used CUDA 11.6 and Pytorch GPU to train all the models where are mentioned in this repository.
<br>

Pleaes refer to the following link to setup Pytorch CUDA locally
https://pytorch.org/get-started/locally/ \
https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
<br>
<br>

![alt model eval](./images/models_eval.png)
We have trained the MobileVit and ResNet5M models in self supervised fashino using self distillation and compared the performances with the state of the art DINOv2.

MobileVit(LowDINO) with 5.5M parameters performs very close to the DINOv2 with 21.5M parameters over KNN evaluation. A similar trend can be observed after fine tuning the models over 10% and 30% of CIFAR 10 datasets, the MobileVit(LowDINO) with 4 times less parameter count performs very close to the State of the Art DINOv2 in classification tasks.

Though these results show promising, more researches have to be done to compare how the model performs in other tasks like image segmentation, etc.
