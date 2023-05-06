import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


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

def Linear(backbone, test_backbone,data_loader_train, data_loader_val):
    """Compute linear evaluation on the dataset.

    Parameters
    ----------
    backbone : timm.models.vision_transformer.VisionTransformer
        Vision transformer. The head should be an identity mapping.

    data_loader_val : torch.utils.data.DataLoader
        Validation dataloader that does not apply any augmentations. Just
        casting to tensor and then normalizing.

    Returns
    -------
    val_accuracy : float
        Validation accuracy.
    """
    test_backbone= test_backbone.load_state_dict(backbone)
    device = next(backbone.parameters()).device

    # Define linear layer
    linear_layer = torch.nn.Linear(test_backbone.num_features, len(data_loader_val.dataset.classes)).to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(linear_layer.parameters())

    # Train linear classifier
    backbone.eval()
    linear_layer.train()
    for epoch in range(1):
        for imgs, labels in data_loader_train:
            imgs = imgs.to(device)
            labels = labels.to(device)
            features = test_backbone(imgs)
            logits = linear_layer(features)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate linear classifier
    test_backbone.eval()
    linear_layer.eval()
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