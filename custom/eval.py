import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from mobile import mobilenet
import os
import pathlib
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder

def compute_knn(backbone, data_loader_train, data_loader_val):
    """Get CLS embeddings and use KNN classifier on them.

    We load all embeddings in memory and use sklearn. Should
    be doable.

    Parameters
    ----------
    backbone : timm.models.vision_transformer.VisionTransformer
        Vision transformer whose head is just an identity
        mapping.

    data_loader_train, data_loader_val : torch.utils.data.DataLoader
        Training and validation dataloader that does not apply any
        augmentations. Just casting to tensor and then normalizing.

    Returns
    -------
    val_accuracy : float
        Validation accuracy.
    """
    device = next(backbone.parameters()).device

    data_loaders = {
        "train": data_loader_train,
        "val": data_loader_val,
    }
    lists = {
        "X_train": [],
        "y_train": [],
        "X_val": [],
        "y_val": [],
    }
    backbone.eval()

    for name, data_loader in data_loaders.items():
        for imgs, y in data_loader:
            imgs = imgs.to(device)
            lists[f"X_{name}"].append(backbone(imgs).detach().cpu().numpy())
            lists[f"y_{name}"].append(y.detach().cpu().numpy())

    arrays = {k: np.concatenate(l) for k, l in lists.items()}

    estimator = KNeighborsClassifier()
    estimator.fit(arrays["X_train"], arrays["y_train"])
    y_val_pred = estimator.predict(arrays["X_val"])

    acc = accuracy_score(arrays["y_val"], y_val_pred)

    return acc

def Linear(backbone,device,data_loader_train, data_loader_val):
    """Compute linear evaluation on the dataset.

    Parameters
    ----------
    backbone : timm.models
       The head should be an identity mapping.

    data_loader_val : torch.utils.data.DataLoader
        Validation dataloader that does not apply any augmentations. Just
        casting to tensor and then normalizing.

    Returns
    -------
    val_accuracy : float
        Validation accuracy.
    """
    num_epochs =1
    if len(os.listdir('Linear_weights'))!=0:
        
        linear_layer = torch.nn.Linear(1024, len(data_loader_val.dataset.classes)).to(device)
        #linear_layer.load_state_dict(torch.load('Linear_weights/mobilevit_linear.pth'))
        test_backbone= backbone
        num_epochs =1
        print(test_backbone.eval())
        print(linear_layer.eval())
    else:
        num_epochs =1
        #train the Linear layer if not pretrained before
        
        test_backbone= backbone
        linear_layer = torch.nn.Linear(test_backbone.num_features, len(data_loader_val.dataset.classes)).to(device)

        device = next(backbone.parameters()).device
        
        # Define loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(linear_layer.parameters(), lr= 0.01)

        # Train linear classifier
        test_backbone.to(device)
        linear_layer.to(device)

        test_backbone.eval()
        linear_layer.train()
        total=0
        for epoch in range(num_epochs):
            running_loss = 0.0
            running_corrects = 0
            i=0
            for imgs, labels in data_loader_train:
                imgs = imgs.to(device)
                labels = labels.to(device)
                features = test_backbone(imgs)
                logits = linear_layer(features)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i=i+1
                total= total+ len(imgs)
                print("current batch",i, total)

                _, preds = torch.max(logits, 1)
                running_loss += loss.item() * imgs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(data_loader_train.dataset)
            epoch_acc = running_corrects.double() / len(data_loader_train.dataset)

            print(f"Epoch {epoch+1}/{num_epochs} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}")

    
        torch.save(linear_layer.state_dict(),f'Linear_weights/mobilevit_linear.pth')

    # Evaluate linear classifier
    test_backbone.eval().to(device)
    linear_layer.eval().to(device)
    y_true = []
    y_pred = []
    with torch.no_grad():
        for imgs, labels in data_loader_val:
            imgs = imgs.to(device)
            labels = labels.to(device)
            features = test_backbone(imgs)
            logits = linear_layer(features)
            preds = logits.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    val_accuracy = accuracy_score(y_true, y_pred)
    print("val accuracy=>",val_accuracy)

    return val_accuracy


def compute_embedding(backbone, data_loader):
    """Compute CLS embedding and prepare for TensorBoard.

    Parameters
    ----------
    backbone : timm.models.vision_transformer.VisionTransformer
        Vision transformer. The head should be an identity mapping.

    data_loader : torch.utils.data.DataLoader
        Validation dataloader that does not apply any augmentations. Just
        casting to tensor and then normalizing.

    Returns
    -------
    embs : torch.Tensor
        Embeddings of shape `(n_samples, out_dim)`.

    imgs : torch.Tensor
        Images of shape `(n_samples, 3, height, width)`.

    labels : list
        List of strings representing the classes.
    """
    device = next(backbone.parameters()).device

    embs_l = []
    imgs_l = []
    labels = []

    for img, y in data_loader:
        img = img.to(device)
        embs_l.append(backbone(img).detach().cpu())
        imgs_l.append(((img * 0.224) + 0.45).cpu())  # undo norm
        labels.extend([data_loader.dataset.classes[i] for i in y.tolist()])

    embs = torch.cat(embs_l, dim=0)
    imgs = torch.cat(imgs_l, dim=0)

    return embs, imgs, labels




if __name__=="__main__":
    transform_plain = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize((224, 224)),
        ]
    )

    img_train_small = "data/imagenette2-320/train"
    img_val_small ="data/imagenette2-320/val"
    path_dataset_train = pathlib.Path(img_train_small)
    path_dataset_val = pathlib.Path(img_val_small)
    dataset_train_plain = ImageFolder(path_dataset_train, transform=transform_plain)
    dataset_train_val = ImageFolder(path_dataset_val, transform=transform_plain)

    data_loader_train_plain = DataLoader(
        dataset_train_plain,
        batch_size=16,
        drop_last=False,
        num_workers=1,
    )
    data_loader_train_plain = DataLoader(
        dataset_train_val,
        batch_size=16,
        drop_last=False,
        num_workers=1,
    )
    backbone= mobilenet()
    backbone =torch.load('Linear_weights/best_model.pth')
    
    #print(backbone.eval())
    val = Linear(backbone,'cuda',data_loader_train_plain,data_loader_train_plain)
    print(val)