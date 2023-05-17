import os
import torch
import sys
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import datetime

#utils contains setting up distributed systems

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


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
        args.world_size = int(os.environ['WORLD_SIZE'])
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

def save_checkpoint(checkpoint_dir, epoch, model, knn_acc,args,time, linear_acc=0, checkpoint_filename="student_model"):
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
        f.write("Epoch {}: KNN accuracy = {}\n".format(epoch, knn_acc))
        f.write("Epoch {}: Linear accuracy = {}\n".format(epoch, linear_acc))
        f.write("Time to comeplete the Epoch = {}\n".format(time))

