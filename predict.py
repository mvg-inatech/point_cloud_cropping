import argparse
from os.path import join, exists
from os import makedirs
import torch
import numpy as np

from common.parser import get_params
from common.metric import IoUEval
from dataset.base_dataset import point_cloud_collate_fn
from dataset.utils import dict_to_device
from dataset.point_cloud_dataset import LargeScaleDataset
from models.model_loader import get_model
from common.parser import yaml_cfg_to_class
from common.io_point_cloud import save_dict_to_laspy


def parse_arguments(parser):
    parser.add_argument(
        "config_dir", type=str, help="dir to config file and model weights"
    )
    parser.add_argument("file_dir", type=str, help="dir to single file")
    parser.add_argument("output_dir", type=str, help="dir to save predictions")
    parser.add_argument(
        "--format",
        type=str,
        choices=[".las", ".ply"],
        default=".ply",
        help="format to use for input data",
    )
    parser.add_argument(
        "--range_center",
        type=float,
        default=5.0,
        help="Specify the range center as single float values",
    )
    parser.add_argument(
        "--validate",
        type=bool,
        default=True,
        help="Whether to run validation after prediction",
    )
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on {}".format(device))

    config = get_params(join(args.config_dir, "config.yaml"))
    dataset_config = yaml_cfg_to_class(
        join(args.config_dir, "config.yaml"),
        "dataset_name",
        "dataset_config",
    )

    model = get_model(config["name"], join(args.config_dir, "config.yaml"))
    model.load_state_dict(
        torch.load(join(args.config_dir, "bird.pt"), map_location=device)
    )
    model = model.to(device)
    model.eval()

    # override dataset config for validation
    dataset_config.val_dir = args.file_dir
    dataset_config.loops = 1
    dataset_config.ending = args.format
    dataset_config.grid_overlay = args.range_center
    dataset_val = LargeScaleDataset(dataset_config, split="val")

    metric_sub = IoUEval(model.nr_classes, device=device)
    metric_full = IoUEval(model.nr_classes, device=device)

    if not exists(args.output_dir):
        makedirs(args.output_dir)

    if args.validate:
        for key, data_dict in dataset_val.full_data_dicts.items():
            if "labels" not in data_dict:
                print(
                    f"Validation requested but no labels found for file {key}. Skipping validation..."
                )
                args.validate = False

    predictions = {}
    for key, data_dict in dataset_val.full_data_dicts.items():
        predictions[key] = np.zeros(
            [data_dict["coords"].shape[0], model.nr_classes], dtype=np.float32
        )

    for item in range(len(dataset_val)):
        data_dict = dataset_val[item]
        data_dict = point_cloud_collate_fn([data_dict])
        data_dict = dict_to_device(data_dict, device)

        with torch.no_grad():
            out = model(data_dict)

        if args.validate:
            metric_sub.add_batch(out.argmax(dim=1), data_dict["labels"])

        idx = data_dict["idx"].cpu().numpy()
        file_name = data_dict["file_name"][0]
        predictions[file_name][idx] += out.softmax(dim=1).cpu().numpy()
        print(f"Done with file {item} of total clouds {len(dataset_val)}", end="\r")

    for key, data_dict in dataset_val.full_data_dicts.items():
        print(f"Saving predictions for file: {key} to dir {args.output_dir}")
        if args.validate:
            metric_full.add_batch(predictions[key].argmax(axis=1), data_dict["labels"])
        pred_max = np.argmax(predictions[key], axis=1)
        data_dict["predictions"] = pred_max
        file_name = key.split("/")[-1].split(".")[0]
        save_dir = join(args.output_dir, f"{file_name}_predictions.las")
        save_dict_to_laspy(data_dict, save_dir)

    if args.validate:
        print("Sub cloud evaluation:")
        metric_sub.print_stats()

        print("Full cloud evaluation:")
        metric_full.print_stats()

    print("HUI")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict on point cloud data")
    args = parse_arguments(parser)
    main(args)
