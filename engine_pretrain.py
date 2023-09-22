# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import tcc_kaggle_code.util.misc as misc
import tcc_kaggle_code.util.lr_sched as lr_sched

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    # Set the model to training mode
    model.train(True)

    # Initialize a metric logger to track training metrics
    metric_logger = misc.MetricLogger(delimiter="  ")

    # Add a meter for learning rate to the metric logger
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    # Create a header for log messages with the current epoch
    header = 'Epoch: [{}]'.format(epoch)

    # Set the frequency for printing training progress
    print_freq = 40

    # Determine the number of accumulation iterations from the arguments
    accum_iter = args.accum_iter

    # Zero out the gradients in the optimizer
    optimizer.zero_grad()

    # Print the log directory if a log writer is provided
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # Loop over batches of data in the data loader
    # for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Adjust the learning rate on a per-iteration basis (instead of per-epoch)
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # Move the data to the specified device
        samples = samples.to(device, non_blocking=True)

        # Use automatic mixed precision (AMP) for training
        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        # Get the loss value as a float
        if loss.numel() > 1:
            loss_value = loss.mean().item()
        else:
            loss_value = loss.item()

        # Check if the loss is finite; exit training if it's not
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Divide the loss by the accumulation iterations
        loss /= accum_iter

        # Use a loss scaler for mixed-precision training
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        # Zero the gradients if it's the last accumulation iteration
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        # Synchronize GPU operations
        torch.cuda.synchronize()

        # Update the metric logger with the loss value
        metric_logger.update(loss=loss_value)

        # Get the current learning rate from the optimizer
        lr = optimizer.param_groups[0]["lr"]

        # Update the metric logger with the current learning rate
        metric_logger.update(lr=lr)

        # Reduce the loss value across all processes (if distributed training)
        loss_value_reduce = misc.all_reduce_mean(loss_value)

        # Log training loss and learning rate if a log writer is provided
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # Synchronize metric logger between all processes (if distributed training)
    metric_logger.synchronize_between_processes()

    # Print averaged training statistics
    print("Averaged stats:", metric_logger)

    # Return averaged metrics in a dictionary
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
