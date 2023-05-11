import os
import torch
import sys
import torchvision
import argparse
import pathlib
import torch.nn as nn
import torch.distributed as dist
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):

    # launched with submitit on a slurm cluster
    if 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpus = int(os.environ["SLURM_GPUS_ON_NODE"])
        args.world_size = os.environ['WORLD_SIZE']
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    #setup_for_distributed(args.rank == 0)

def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False



def main(args):
    init_distributed_mode(args)
    transform= transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize((224, 224)),
        ]
    )
    if args.rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

    args.local_rank = args.rank  - args.gpus * (args.rank // args.gpus)
    torch.cuda.set_device(args.local_rank)
    device = 'cuda'

    img_val_small ="/scratch/sp7238/DL/LowDINO/custom/data/imagenette2-320/val"
    path_dataset_val = pathlib.Path(img_val_small)
    dataset_train_val = ImageFolder(path_dataset_val, transform=transform)
    dataset = ImageFolder('/scratch/sp7238/DL/LowDINO/custom/data/imagenette2-320/train', transform=transform)
    sampler = torch.utils.data.ditributed.DistributedSampler(dataset, shuffle=True)
    student = torchvision.models.resnet50()
    student.fc.out_features=10


    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.local_rank])
    if has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
    
    lr = 0.01
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
     )
    data_loader_train_plain = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=32,
        drop_last=False,
        num_workers=1,
    )
    num_epochs=5
    for epoch in range(num_epochs):
            running_loss = 0.0
            running_corrects = 0
            i=0
            for imgs, labels in data_loader_train_plain:
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = student(imgs)
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

            epoch_loss = running_loss / len(data_loader_train_plain.dataset)
            epoch_acc = running_corrects.double() / len(data_loader_train_plain.dataset)
            print(f"Epoch {epoch+1}/{num_epochs} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}")


  
if __name__=="__main__":
        parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
        parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                            help='input batch size for training (default: 64)')
        args = parser.parse_args() 
        main(args)
    #val = Linear(backbone,'cuda',data_loader_train_plain,data_loader_train_plain)
    
