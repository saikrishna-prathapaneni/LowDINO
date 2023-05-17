import pathlib
import os
import torch
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import time
from eval import  compute_knn
from Augmentation import DataAugmentation
from model import Head, DinoLoss, MultiCrop
from mobile import mobilenet
from utils import save_checkpoint
import colossalai
from colossalai.core import global_context as gpc
from colossalai.utils import get_dataloader
from torchvision import transforms
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.logging import get_dist_logger

checkpoint_dir ="checkpoints"



def main(args):

    colossalai.launch_from_torch(config='./config.py')

    logger = get_dist_logger()

   
    print(vars(args))
    # Parameters

    IMAGENET1K_TRAIN ="/vast/work/public/ml-datasets/imagenet/train"
    IMAGENET1K_TEST = "/scratch/sp7238/DL/data/val/val_2"

    img_train_small = "/scratch/sp7238/DL/LowDINO/custom/data/imagenette2-320/train"
    img_val_small ="/scratch/sp7238/DL/LowDINO/custom/data/imagenette2-320/val"

    # change the path to train and test for train and test

    dim = 640 # adjust the dimension according to the model used #for resnet dim =512
    path_dataset_train = pathlib.Path(IMAGENET1K_TRAIN)
    path_dataset_val = pathlib.Path(IMAGENET1K_TEST)
 

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
    dataset_val_plain = ImageFolder(path_dataset_val, transform=transform_plain)
    dataset_train_plain_test = ImageFolder(img_train_small, transform=transform_plain)
    dataset_val_plain_test = ImageFolder(img_train_small, transform=transform_plain)


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


    colossalai_train_dataloader = get_dataloader(
        dataset=dataset_train_aug,
        shuffle=True,
        batch_size=gpc.config.BATCH_SIZE,
        pin_memory=True,
    )
    colossalai_test_dataloader = get_dataloader(
        dataset=dataset_val_plain,
        add_sampler=False,
        batch_size=gpc.config.BATCH_SIZE,
        pin_memory=True,
    )
 

    
    student_vit = mobilenet('mobilevit_s',pretrained=args.pretrained)
    teacher_vit = mobilenet('mobilevit_s',pretrained=args.pretrained)


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
    steps_per_epoch = len(colossalai_train_dataloader)
    lr_scheduler = CosineAnnealingWarmupLR(optimizer=optimizer,
                                           total_steps=gpc.config.NUM_EPOCHS * steps_per_epoch,
                                           warmup_steps=gpc.config.WARMUP_EPOCHS * steps_per_epoch)

    student_engine, colossalai_train_dataloader, colossalai_test_dataloader, _ = colossalai.initialize(
        student,
        optimizer,
       loss_inst,
        colossalai_train_dataloader,
        colossalai_test_dataloader,
    )
    teacher.to(device)

    # Training loop

    n_steps = 0

    for e in range(gpc.config.NUM_EPOCHS):
        epoch_start_time = time.perf_counter()
        print("currently running epoch =>", e)
        student_engine.train()
        for i, (images, _) in enumerate(colossalai_train_dataloader):
            images = [img.to(device) for img in images]

            teacher_output = teacher(images[:2])
            student_output = student_engine(images)

            loss = student_engine.criterion(student_output, teacher_output)

            student_engine.zero_grad()
            student_engine.backward(loss)
            #clip_gradients(student_engine, args.clip_grad)
            student_engine.step()

            with torch.no_grad():
                for student_ps, teacher_ps in zip(
                    student_engine.model.parameters(), teacher.parameters()
                ):
                    teacher_ps.data.mul_(args.momentum_teacher)
                    teacher_ps.data.add_(
                        (1 - args.momentum_teacher) * student_ps.detach().data
                    )

            print(f"train_loss epoch number {e}", loss)
            n_steps += 1
        lr_scheduler.step()
        epoch_end_time = time.perf_counter()
        epoch_time = epoch_end_time - epoch_start_time
        print("time taken to comeplete one epoch",epoch_time)

        if e % args.logging_freq == 0:
                student_engine.eval()

                knn_acc = compute_knn(
                    student.backbone,
                    data_loader_train_test,
                    data_loader_val_test,
                )
        
                try:
                    save_checkpoint(checkpoint_dir=checkpoint_dir,
                                    epoch=e,
                                    model=student,
                                    args=args,
                                    knn_acc=knn_acc,
                                    time= epoch_time
                                    )
                except:
                    print('Directory Exists')
                print("knn_accuracy => ",knn_acc)
                student.train()


if __name__ == "__main__":
    parser = colossalai.get_default_parser()
    parser.add_argument('--use_trainer', action='store_true', help='whether to use trainer')
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

