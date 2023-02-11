# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Fine-tune a BiT model on some downstream dataset."""
#!/usr/bin/env python3
# coding: utf-8
import os
from os.path import join as pjoin  # pylint: disable=g-importing-member
import time

import numpy as np
import torch
import torchvision as tv
from torch.utils import data

from .lbtoolbox import Uninterrupt, Chrono
from .lr_schduler import (
    get_lr,
    get_mixup,
)


def topk(output, target, ks=(1,)):
    """Returns one boolean vector for each k, whether the target is within the output's top-k."""
    _, pred = output.topk(max(ks), 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].max(0)[0] for k in ks]


def recycle(iterable):
    """Variant of itertools.cycle that does not save iterates."""
    while True:
        for i in iterable:
            yield i



def run_eval(model, data_loader, device, chrono, logger, step, ologger=None):
    # switch to evaluate mode
    model.eval()
    logger.info("Running validation...")
    logger.flush()

    all_c, all_top1, all_top5, acc = [], [], [], []
    end = time.time()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # measure data loading time
            chrono._done("eval load", time.time() - end)

            # compute output, measure accuracy and record loss.
            with chrono.measure("eval fprop"):
                logits = model(x)
                c = torch.nn.CrossEntropyLoss(reduction="none")(logits, y)
                top1, top5 = topk(logits, y, ks=(1, 5))
                all_c.extend(c.cpu())  # Also ensures a sync point.
                all_top1.extend(top1.cpu())
                all_top5.extend(top5.cpu())

        # measure elapsed time
        end = time.time()

    model.train()
    logger.info(
        f"Validation@{step} loss {np.mean(all_c):.5f}, "
        f"top1 {np.mean(all_top1):.2%}, "
        f"top5 {np.mean(all_top5):.2%}"
    )
    logger.flush()

    if ologger is not None:
        ologger.add_scalar("Loss/Val", np.mean(all_c), step)
        ologger.add_scalar("Top1/Val", np.mean(all_top1), step)
        ologger.add_scalar("Top5/Val", np.mean(all_top5), step)

    return all_c, all_top1, all_top5


def mixup_data(x, y, l):
    """Returns mixed inputs, pairs of targets, and lambda"""
    indices = torch.randperm(x.shape[0]).to(x.device)

    mixed_x = l * x + (1 - l) * x[indices]
    y_a, y_b = y, y[indices]
    return mixed_x, y_a, y_b


def mixup_criterion(criterion, pred, y_a, y_b, l):
    return l * criterion(pred, y_a) + (1 - l) * criterion(pred, y_b)


def train(
    model: torch.nn.Module,
    train_loader: data.Dataset,
    valid_loader: data.Dataset,
    train_set_size: int,
    save: bool,
    save_path: str,
    batch_split: int,
    base_lr: float,
    eval_every: int,
    tensorboardlogger=None,
):
    logger = tensorboardlogger
    logger.__setattr__("info", print)

    # Lets cuDNN benchmark conv implementations and choose the fastest.
    # Only good if sizes stay the same within the main loop!
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Going to train on {device}")


    # model.load_from(np.load(f"{model_name}.npz"))

    logger.info("Moving model onto all GPUs")
    model = torch.nn.DataParallel(model)

    # Optionally resume from a checkpoint.
    # Load it to CPU first as we'll move the model to GPU later.
    # This way, we save a little bit of GPU memory when loading.
    step = 0

    # Note: no weight-decay!
    optim = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    # Resume fine-tuning if we find a saved model.
    savename = save_path
    try:
        logger.info(f"Model will be saved in '{savename}'")
        checkpoint = torch.load(savename, map_location="cpu")
        logger.info(f"Found saved model to resume from at '{savename}'")

        step = checkpoint["step"]
        model.load_state_dict(checkpoint["model"])
        optim.load_state_dict(checkpoint["optim"])
        logger.info(f"Resumed at step {step}")
    except FileNotFoundError:
        logger.info("Fine-tuning from BiT")

    model = model.to(device)
    optim.zero_grad()

    model.train()
    mixup = get_mixup(train_set_size)
    cri = torch.nn.CrossEntropyLoss().to(device)

    logger.info("Starting training!")
    chrono = Chrono()
    accum_steps = 0
    mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1
    end = time.time()

    # reset intrupt at first:

    with Uninterrupt() as u:
        for x, y in recycle(train_loader):
            # measure data loading time, which is spent in the `for` statement.
            chrono._done("load", time.time() - end)

            if u.interrupted:
                break

            # Schedule sending to GPU(s)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            lr = get_lr(step, train_set_size, base_lr)

            if lr is None:
                print("LR NOT FOUND! ERROR!")
                break
            for param_group in optim.param_groups:
                param_group["lr"] = lr

            if mixup > 0.0:
                x, y_a, y_b = mixup_data(x, y, mixup_l)

            # compute output
            with chrono.measure("fprop"):
                logits = model(x)
                if mixup > 0.0:
                    c = mixup_criterion(cri, logits, y_a, y_b, mixup_l)
                else:
                    c = cri(logits, y)
                c_num = float(c.data.cpu().numpy())  # Also ensures a sync point.

            # Accumulate grads
            with chrono.measure("grads"):
                (c / batch_split).backward()
                accum_steps += 1

            accstep = f" ({accum_steps}/{batch_split})" if batch_split > 1 else ""
            logger.info(
                f"[step {step}{accstep}]: loss={c_num:.9f} (lr={lr:.1e})"
            )  # pylint: disable=logging-format-interpolation
            # logger.flush()

            if tensorboardlogger is not None:
                tensorboardlogger.add_scalar("Loss/train", c_num, step)
                tensorboardlogger.add_scalar("Learning rate", lr, step)

            # Update params
            if accum_steps == batch_split:
                with chrono.measure("update"):
                    optim.step()
                    optim.zero_grad()
                step += 1
                accum_steps = 0
                # Sample new mixup ratio for next batch
                mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1

                # Run evaluation and save the model.
                if eval_every and step % eval_every == 0:
                    run_eval(model, valid_loader, device, chrono, logger, step, tensorboardlogger)
                    if save:
                        torch.save(
                            {
                                "step": step,
                                "model": model.state_dict(),
                                "optim": optim.state_dict(),
                            },
                            savename,
                        )

            end = time.time()

        # Final eval at end of training.
        run_eval(model, valid_loader, device, chrono, logger, step="end")
        u.release()


    logger.info(f"Timings:\n{chrono}")
