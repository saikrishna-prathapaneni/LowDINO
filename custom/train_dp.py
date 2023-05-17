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
from utils import has_batchnorms
from eval import  compute_knn
from Augmentation import DataAugmentation
from model import Head, DinoLoss, MultiCrop, clip_gradients
from mobile import mobilenet
from utils import save_checkpoint


#data parallel scripting for the model


checkpoint_dir ="checkpoints_DP"

def main(args):
  
    dim =640 # change this parameter according to the model backbone output you are using for Mobilevit it is 640
   
    device = 'cuda'

    
    IMAGENET1K_TRAIN ="/vast/work/public/ml-datasets/imagenet/train"
    IMAGENET1K_TEST = "/scratch/sp7238/DL/data/val/val_2"

    img_train_small = "/scratch/sp7238/DL/LowDINO/custom/data/imagenette2-320/train"
    img_val_small ="/scratch/sp7238/DL/LowDINO/custom/data/imagenette2-320/val"

    path_dataset_train = pathlib.Path(IMAGENET1K_TRAIN)
    path_dataset_val = pathlib.Path(IMAGENET1K_TEST)
 
    img_train_small = pathlib.Path(img_train_small)
    img_val_small = pathlib.Path(img_val_small)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    
    transform_aug = DataAugmentation(size=224, n_local_crops=args.n_crops - 2)
    transform_plain = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize((224, 224)),
        ]
    )

    dataset_train_aug = ImageFolder(path_dataset_train, transform=transform_aug)
    #dataset_train_plain = ImageFolder(path_dataset_train, transform=transform_plain)
    dataset_val_plain = ImageFolder(path_dataset_val, transform=transform_plain)
    dataset_train_plain_test = ImageFolder(img_train_small, transform=transform_plain)
    dataset_val_plain_test = ImageFolder(img_val_small, transform=transform_plain)


    data_loader_train_test = DataLoader(
        dataset_train_plain_test,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
        pin_memory=True,
    )
    data_loader_val_test = DataLoader(
        dataset_val_plain_test,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
        pin_memory=True,
    )


    colossalai_train_dataloader = DataLoader(
        dataset_train_aug,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
        pin_memory=True
    )
    colossalai_test_dataloader = DataLoader(
        dataset=dataset_val_plain,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
        pin_memory=True
    )
 

    #teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.local_rank])
    student_vit = mobilenet('mobilevit_s',pretrained=args.pretrained)
    teacher_vit = mobilenet('mobilevit_s',pretrained=args.pretrained)
    test_student= mobilenet('mobilevit_s',pretrained=args.pretrained)

    student = MultiCrop(
        student_vit,
        Head(
            dim,
            args.out_dim,
            norm_last_layer=args.norm_last_layer,
        ),
    )
    teacher = MultiCrop(teacher_vit, Head(dim, args.out_dim))
   
    teacher.load_state_dict(student.state_dict())

    for p in teacher.parameters():
        p.requires_grad = False\
    
    student = nn.DataParallel(student, device_ids=args.device_ids)
    teacher = nn.DataParallel(teacher,device_ids=args.device_ids)
   
    if has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
    
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    lr = 0.0005 * args.batch_size / 256
   
    loss_inst = DinoLoss(
        args.out_dim,
        teacher_temp=args.teacher_temp,
        student_temp=args.student_temp,
    ).to(device)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
     )
    

    for e in range(args.n_epochs):
        print("currently running epoch =>", e)
        for i, (images, _) in enumerate(colossalai_train_dataloader):
            images = [img.to(device) for img in images]

            teacher_output = teacher(images[:2])
            student_output = student(images)

            loss = loss_inst(student_output, teacher_output)

            optimizer.zero_grad()
            loss.backward()
            clip_gradients(student, args.clip_grad)
            optimizer.step()

            with torch.no_grad():
                for student_ps, teacher_ps in zip(
                    student.parameters(), teacher.parameters()
                ):
                    teacher_ps.data.mul_(args.momentum_teacher)
                    teacher_ps.data.add_(
                        (1 - args.momentum_teacher) * student_ps.detach().data
                    )

            print(f"train_loss epoch number {e}", loss)
            

        if e % args.logging_freq == 0:
                student.eval()
                knn_acc = compute_knn(
                    student.module.backbone,
                    data_loader_train_test,
                    data_loader_val_test,
                )

                save_checkpoint(checkpoint_dir=checkpoint_dir,
                                epoch=e,
                                model=student,
                                time=0,
                                args=args,
                                knn_acc=knn_acc
                                )
        
                student.train()


  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DINO training CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-b", "--batch-size", type=int, default=1024)
    parser.add_argument("-l", "--logging-freq", type=int, default=1)
    parser.add_argument("--momentum-teacher", type=int, default=0.9995)
    parser.add_argument("-c", "--n-crops", type=int, default=4)
    parser.add_argument("-e", "--n-epochs", type=int, default=50)
    parser.add_argument("-o", "--out-dim", type=int, default=1024)
    parser.add_argument("--clip-grad", type=float, default=2.0)
    parser.add_argument("--norm-last-layer", action="store_true")
    parser.add_argument("--batch-size-eval", type=int, default=8)
    parser.add_argument("--teacher-temp", type=float, default=0.04)
    parser.add_argument("--student-temp", type=float, default=0.1)
    parser.add_argument("-d", "--device-ids", type=list, default=[0,1])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("-w", "--weight-decay", type=float, default=0.4)

    args = parser.parse_args()
    main(args)
    #val = Linear(backbone,'cuda',data_loader_train_plain,data_loader_train_plain)
    
