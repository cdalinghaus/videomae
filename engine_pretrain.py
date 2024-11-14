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

import util.misc as misc
import util.lr_sched as lr_sched

import matplotlib.pyplot as plt

from PIL import Image

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        #print(samples.shape)
        samples = samples.to(device, non_blocking=True)

        img_size = samples.shape[-1]
        batch_size = samples.shape[0]
        # Step 1: Permute to bring the frames dimension next to height and width
        samples = samples.permute(0, 2, 1, 3, 4)  # shape [64, 3, 16, 128, 128]

        # Step 2: Reshape frames into a 4x4 grid
        samples = samples.reshape(batch_size, 3, 4, 4, img_size, img_size)  # shape [64, 3, 4, 4, 128, 128]

        # Step 3: Permute to arrange the grid along height and width
        samples = samples.permute(0, 1, 2, 4, 3, 5)  # shape [64, 3, 4, 128, 4, 128]

        # Step 4: Reshape to combine the grid into one large image
        samples = samples.reshape(batch_size, 3, 4 * img_size, 4 * img_size)  # shape [64, 3, 512, 512]

        with torch.cuda.amp.autocast():
            loss, _pred, _mask = model(samples, mask_ratio=args.mask_ratio)

        if data_iter_step % 250 == 0:
            unpatched = model.unpatchify(_pred).float()
            combined = torch.cat([unpatched[0].detach().cpu().T, samples[0].detach().cpu().T], axis=0)

            log_writer.add_image(f'Example reconstruction from train set', combined.detach().cpu().T, epoch)

            del combined
            del unpatched

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}