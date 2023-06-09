import argparse
import json
import pathlib
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
import datetime
from eval import  compute_knn,Linear
from Augmentation import DataAugmentation
from model import Head, DinoLoss, MultiCrop, clip_gradients
from mobile import mobilenet
from utils import save_checkpoint

checkpoint_dir ="checkpoints"



def main(args):
   
    # Parameters

    IMAGENET1K_TRAIN ="/vast/work/public/ml-datasets/imagenet/train"
    IMAGENET1K_TEST = "/scratch/sp7238/DL/data/val/val_2"

    img_train_small = "/scratch/sp7238/DL/LowDINO/custom/data/imagenette2-320/train"
    img_val_small ="/scratch/sp7238/DL/LowDINO/custom/data/imagenette2-320/val"

    vit_name, dim = "vit_small_patch16_224", 640
    path_dataset_train = pathlib.Path(img_train_small)
    path_dataset_val = pathlib.Path(img_val_small)
 

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    n_workers = 2



    transform_aug = DataAugmentation(size=224, n_local_crops=args.n_crops - 2)
    transform_plain = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize((224, 224)),
        ]
    )

    dataset_train_aug = ImageFolder(path_dataset_train, transform=transform_aug)
    dataset_train_plain = ImageFolder(path_dataset_train, transform=transform_plain)
    dataset_val_plain = ImageFolder(path_dataset_val, transform=transform_plain)


    data_loader_train_aug = DataLoader(
        dataset_train_aug,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
    )
    data_loader_train_plain = DataLoader(
        dataset_train_plain,
        batch_size=args.batch_size_eval,
        drop_last=False,
        num_workers=n_workers,
    )
    data_loader_val_plain = DataLoader(
        dataset_val_plain,
        batch_size=args.batch_size_eval,
        drop_last=False,
        num_workers=n_workers,
    )
   
 
    # Neural network related
    # student_vit = timm.create_model(vit_name, pretrained=args.pretrained)
    # teacher_vit = timm.create_model(vit_name, pretrained=args.pretrained)
    
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
    student, teacher = student.to(device), teacher.to(device)

    teacher.load_state_dict(student.state_dict())

    for p in teacher.parameters():
        p.requires_grad = False

    # Loss related
    loss_inst = DinoLoss(
        args.out_dim,
        teacher_temp=args.teacher_temp,
        student_temp=args.student_temp,
    ).to(device)
    
    lr = 0.0005 * args.batch_size / 256
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
    )

    # Training loop
    n_batches = len(dataset_train_aug) // args.batch_size
    best_acc = 0
    n_steps = 0

    for e in range(args.n_epochs):
        print("currently running epoch =>", e)
        for i, (images, _) in enumerate(data_loader_train_aug):
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
            n_steps += 1

        if e % args.logging_freq == 0:
                student.eval()

                knn_acc = compute_knn(
                    student.backbone,
                    data_loader_train_plain,
                    data_loader_val_plain,
                )
                # linear_acc = Linear(
                #     student.backbone,
                #     data_loader_train_plain,
                #     data_loader_val_plain,
                # )
                save_checkpoint(checkpoint_dir=checkpoint_dir,
                                epoch=e,
                                model=student,
                                args=args,
                                knn_acc=knn_acc
                               #linear_acc=linear_acc,
                                )
                print("knn_accuracy => ",knn_acc)
                # if knn_acc > best_acc:
                #     torch.save(student, "best_model.pth")
                #     best_acc = knn_acc
                #     print("best_accuracy knn => ",best_acc)
                student.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DINO training CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-b", "--batch-size", type=int, default=16)
    parser.add_argument("-l", "--logging-freq", type=int, default=1)
    parser.add_argument("--momentum-teacher", type=int, default=0.9995)
    parser.add_argument("-c", "--n-crops", type=int, default=8)
    parser.add_argument("-e", "--n-epochs", type=int, default=50)
    parser.add_argument("-o", "--out-dim", type=int, default=1024)
    parser.add_argument("-t", "--tensorboard-dir", type=str, default="logs")
    parser.add_argument("--clip-grad", type=float, default=2.0)
    parser.add_argument("--norm-last-layer", action="store_true")
    parser.add_argument("--batch-size-eval", type=int, default=8)
    parser.add_argument("--teacher-temp", type=float, default=0.04)
    parser.add_argument("--student-temp", type=float, default=0.1)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("-w", "--weight-decay", type=float, default=0.4)

    args = parser.parse_args()
    main(args)

