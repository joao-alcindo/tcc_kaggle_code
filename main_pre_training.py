import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path


import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets



import timm
# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory




import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.custom_dataset import CustomDataset
import util.transform_npy as transform_npy


import models_mae

from engine_pretrain import train_one_epoch


import pdb



def main(args):
    # Initialize distributed mode if needed
    misc.init_distributed_mode(args)  # Initialize distributed training if required

    # Print job directory and arguments
    # Get the current working directory
    current_dir = os.getcwd()
    print('job dir:', current_dir)

    print("{}".format(args).replace(', ', ',\n'))

    # Set the device for training (e.g., 'cuda' or 'cpu')
    device = torch.device(args.device)

    # Fix the random seed for reproducibility
    seed = args.seed + misc.get_rank()  # Combine provided seed and distributed rank
    torch.manual_seed(seed)  # Set PyTorch random seed
    np.random.seed(seed)  # Set NumPy random seed

    cudnn.benchmark = True  # Enable CuDNN benchmark mode for optimized performance


    transform_train = transforms.Compose([
        transform_npy.ResizeNpyWithPadding((args.input_size, args.input_size)),
        transform_npy.RandomHorizontalFlipNpy(),
        transform_npy.RandomRotationNpy(degrees=(-15, 15)),
        transforms.Lambda(lambda data: data.copy()),  # Copy the data to avoid grad error
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
])
    
    # Create a training dataset using the defined transformations
    dataset_train = CustomDataset(data_path=os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    # Configure data sampler for distributed training (if applicable)
    if args.distributed:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    # Set up logging writer for TensorBoard
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # Create data loader for training dataset
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # Define the neural model using the specified architecture
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    # Move the model to the specified device
    model.to(device)

    # Calculate effective batch size for training
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    # Calculate learning rate based on batch size and base learning rate
    if args.lr is None:  # If only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    # Print learning rate and other training settings
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # Set up DistributedDataParallel if using distributed training
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    


    # Set weight decay for bias and norm layers following timm's recommendation
    param_groups = []
    for name, param in model_without_ddp.named_parameters():
        if 'bn' not in name:  # Exclude batch normalization layers from weight decay
            param_groups.append(param)

    #pdb.set_trace()

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    
    #pdb.set_trace()

    print(optimizer)
    # Create a loss scaler for mixed-precision training
    loss_scaler = NativeScaler()

    # Load model checkpoint and optimizer state if available
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Start training loop for specified number of epochs
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # Set epoch for distributed training
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        # Perform one epoch of training and get training statistics
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        
        # Save model checkpoint and statistics periodically
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        # Prepare log statistics for logging
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        # Write log statistics to file if applicable
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    # Calculate total training time and print
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))