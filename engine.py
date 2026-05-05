import torch

# from torch.cuda.amp import autocast, GradScaler
import numpy as np
from common.metric import IoUEval
from dataset.utils import dict_to_device
from models.loss import LossConfig


def clear_console_prints():
    ctrl = "                    "
    print(ctrl + ctrl + ctrl + ctrl, end="\r")


def train_epoch(
    model,
    optimizer,
    criterion,
    loader,
    device,
    scheduler,
    accumulation_steps=1,
    loss_weight_factor=1,
    clip_gradient=3,
):
    model = model.to(device)
    model.train()

    # scaler = GradScaler()

    loss_list = []

    for i, data_dict in enumerate(loader):
        data_dict = dict_to_device(data_dict, device)
        # with autocast():
        pred = model(data_dict)
        loss = criterion(
            pred,
            data_dict["labels"],
            LossConfig(data_dict["probabilities"], factor=loss_weight_factor),
        )
        loss_value = loss.item()
        loss = loss / accumulation_steps
        # scaler.scale(loss).backward()
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            if clip_gradient is not None:
                # scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradient)
            # scaler.step(optimizer)
            # scaler.update()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        loss_list.append(loss_value)

        print(
            ("TRAIN: input {}  of {} --> Loss: {}").format(
                i, len(loader), np.round(np.mean(loss_list), 4)
            ),
            end="\r",
        )

    # final step if dataset size is not divisible by accumulation_steps
    if (i + 1) % accumulation_steps != 0:
        if clip_gradient is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradient)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    clear_console_prints()

    return loss_list


def eval_epoch(model, loader, device, n_classes):
    model = model.to(device)
    model.eval()

    iou_eval = IoUEval(n_classes, device)

    for i, data_dict in enumerate(loader):
        with torch.no_grad():
            data_dict = dict_to_device(data_dict, device)
            pred = model(data_dict)
            pred_max = torch.argmax(pred, 1)

            iou_eval.add_batch(pred_max, data_dict["labels"])

            m_accuracy = np.round(iou_eval.get_Acc().cpu() * 100, 4)
            m_jaccard = np.round(iou_eval.get_mIoU().cpu() * 100, 4)

            print(
                ("EVAL: input {} of {}\t --> mIoU: {} pwAcc: {}").format(
                    i, len(loader), m_jaccard, m_accuracy
                ),
                end="\r",
            )
    clear_console_prints()

    return iou_eval
