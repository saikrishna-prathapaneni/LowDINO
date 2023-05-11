import os
import torch
import sys
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
import datetime
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
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
        args.gpu = args.rank % torch.cuda.device_count()
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
    print('| distributed init (rank {})'.format(
        args.rank), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)

def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False

def save_checkpoint(checkpoint_dir, epoch, model, knn_acc,args, linear_acc=0, checkpoint_filename="student_model"):
    now = datetime.datetime.now()
    iteration_dir = now.strftime("%Y-%m-%d_%H-%M-%S")+'_epoch_'+str(epoch)
    print(iteration_dir)
    os.makedirs(os.path.join(checkpoint_dir, iteration_dir))

    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        
        'knn_accuracy': knn_acc,
        'linear_acc':linear_acc
    }

    #print(checkpoint_data)
    checkpoint_path = os.path.join(checkpoint_dir, iteration_dir, checkpoint_filename + "_epoch{}.pth".format(epoch))
    torch.save(checkpoint_data, checkpoint_path)
    print(checkpoint_path)
    # Save the args and accuracy to a separate file
    args_filename = os.path.join(checkpoint_dir, iteration_dir, "args.txt")
    with open(args_filename, "w") as f:
        for arg in vars(args):
            f.write("{}: {}\n".format(arg, getattr(args, arg)))
    accuracy_filename = os.path.join(checkpoint_dir, iteration_dir, "accuracy.txt")
    print(accuracy_filename)
    with open(accuracy_filename, "a") as f:
        f.write("Epoch {}: accuracy = {}\n".format(epoch, knn_acc))
        f.write("Epoch {}: accuracy = {}\n".format(epoch, linear_acc))
