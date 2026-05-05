import argparse
from os.path import join, exists
from os import makedirs
from datetime import datetime
import yaml
import random

import numpy as np
import torch

from models.model_loader import get_model
from models.loss import get_loss_function
from common.parser import get_params
from engine import train_epoch, eval_epoch
from common.parser import yaml_cfg_to_class
from dataset.point_cloud_dataset import LargeScaleDataset
from dataset.base_dataset import point_cloud_collate_fn


def parse_arguments(parser):
    parser.add_argument("config_dir", type=str, help="dir to config file")
    args = parser.parse_args()
    return args


def get_learning_rate(optimizer):
    learning_rate = 0
    for param_group in optimizer.param_groups:
        learning_rate = param_group["lr"]
    return learning_rate


def create_directory(path):
    now = datetime.now()
    now = now.strftime("%d:%m:%Y-%H:%M")
    path = join(path, now)
    if not exists(path):
        makedirs(path)
    return path


def save_config(config, path):
    file = open(path + "/config.yaml", "w")
    yaml.safe_dump(config, file)


def save_model(model, path):
    name = "bird.pt"
    file_name = join(path, name)
    torch.save(model.state_dict(), file_name)


def build_optimizer(model, lr_list, weight_decay, opti_group):
    """
    Build the optimizer for the model. Splits the parameters into len(lr_list) groups
    based on the opti_group names.
    Args:
        model: The model to optimize.
        lr_list: List of learning rates for each group.
        weight_decay: Weight decay for the optimizer.
        opti_group: List of strings indicating which parameters belong to which group.
            Has to be of length len(lr_list) - 1.
    Returns:
        optimizer: The built optimizer.
    """
    regular_params = []
    other_params = dict()
    for name in opti_group:
        other_params[name] = []

    for name, param in model.named_parameters():
        in_group = False
        if not param.requires_grad:
            continue

        for group in opti_group:
            if group in name.lower():
                other_params[group].append(param)
                in_group = True

        if not in_group:
            regular_params.append(param)

    param_groups = []
    default_lr = lr_list[0]
    param_groups.append({"params": regular_params, "lr": default_lr})
    for i, name in enumerate(opti_group):
        block_lr = lr_list[i + 1]
        param_groups.append({"params": other_params[name], "lr": block_lr})

    for group in param_groups:
        print(
            f"Optimizer group with lr {group['lr']} and {len(group['params'])} parameters."
        )

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    return optimizer


def worker_init_fn(worker_id):
    """Set different random seeds for each worker"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def main(args):
    config = get_params(args.config_dir)
    model = get_model(config["name"], args.config_dir)

    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on {}".format(device))

    path = create_directory(config["save_dir"])
    save_config(config, path)

    dataset_config = yaml_cfg_to_class(
        args.config_dir,
        "dataset_name",
        "dataset_config",
    )

    print("*" * 50)
    print(
        f"Using lambda: {dataset_config.lambda_p} with vs {dataset_config.voxel_size} and method {dataset_config.sub_cloud_method}"
    )
    print("*" * 50)

    dataset_train = LargeScaleDataset(dataset_config, split="train")
    dataset_config.loops = 1  # evaluation only iterates over the dataset once
    dataset_val = LargeScaleDataset(dataset_config, split="val")
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config["train"]["bs"],
        num_workers=config["train"]["nr_workers"],
        persistent_workers=True if config["train"]["nr_workers"] > 0 else False,
        worker_init_fn=worker_init_fn,
        shuffle=True,
        pin_memory=True,
        collate_fn=point_cloud_collate_fn,
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        collate_fn=point_cloud_collate_fn,
    )

    # if only single lr given, no parameter groupsx
    if len(config["train"]["max_lr"]) == 1:
        config["train"]["opti_group"] = []
    assert (
        len(config["train"]["max_lr"]) == len(config["train"]["opti_group"]) + 1
    ), "Length of max_lr must be one more than length of opti_group"
    optimizer = build_optimizer(
        model,
        config["train"]["max_lr"],
        config["train"]["wd"],
        config["train"]["opti_group"],
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["train"]["max_lr"],
        steps_per_epoch=len(dataloader_train) // config["train"]["accumulation_steps"],
        epochs=config["train"]["ep"],
        pct_start=0.05,
        div_factor=10,
        final_div_factor=1000,
    )

    criterion = get_loss_function(
        config["loss"]["name"],
        label_smoothing=config["loss"]["label_smoothing"],
        weight=config["loss"]["weight"],
        ignore_index=config["loss"]["ignore_index"],
    )

    best_val_mIoU = 0
    for ep in range(config["train"]["ep"]):
        print(
            "Starting with Epoch: {} and LR: {}".format(
                ep, get_learning_rate(optimizer)
            )
        )

        loss_list = train_epoch(
            model,
            optimizer,
            criterion,
            dataloader_train,
            device,
            scheduler,
            config["train"]["accumulation_steps"],
            config["loss"]["weight_factor"],
            config["train"]["clip_gradient"],
        )
        print("TRAIN\t --> mLoss: {}".format(np.mean(loss_list)))

        iou_metric = eval_epoch(
            model,
            dataloader_val,
            device,
            config["model_config"]["nr_classes"],
        )

        mIoU = iou_metric.get_mIoU().cpu().numpy()
        IoU = iou_metric.get_IoU().cpu().numpy()

        print("VAL\t --> mIoU: {}".format(mIoU))
        if mIoU > best_val_mIoU:
            best_val_mIoU = mIoU
            print("Best mean iou in validation set so far, save model!")
            print(IoU)
            save_model(model, path)
            iou_metric.save_conf_matrix(path)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="training",
        description="run training for specified model and data",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    args = parse_arguments(parser)
    main(args)
