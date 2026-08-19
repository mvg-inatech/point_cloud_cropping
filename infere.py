import argparse
from os.path import exists, join
from os import makedirs

import numpy as np
import torch

from common.io_point_cloud import save_dict_to_laspy
from common.parser import get_params, yaml_cfg_to_class
from dataset.base_dataset import point_cloud_collate_fn
from dataset.point_cloud_dataset import LargeScaleDataset
from dataset.utils import dict_to_device
from models.model_loader import get_model


def parse_arguments(parser):
    parser.add_argument(
        "config_dir", type=str, help="dir containing config.yaml and bird.pt"
    )
    parser.add_argument("file_dir", type=str, help="dir with input files")
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
        help="range center used for sub-cloud generation",
    )
    return parser.parse_args()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    config_path = join(args.config_dir, "config.yaml")
    weights_path = join(args.config_dir, "bird.pt")

    config = get_params(config_path)
    dataset_config = yaml_cfg_to_class(config_path, "dataset_name", "dataset_config")

    model = get_model(config["name"], config_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    # override dataset config for inference
    dataset_config.val_dir = args.file_dir
    dataset_config.loops = 1
    dataset_config.ending = args.format
    dataset_config.grid_overlay = args.range_center
    dataset_val = LargeScaleDataset(dataset_config, split="val")

    if not exists(args.output_dir):
        makedirs(args.output_dir)

    predictions = {}
    for key, data_dict in dataset_val.full_data_dicts.items():
        predictions[key] = np.zeros(
            (data_dict["coords"].shape[0], model.nr_classes), dtype=np.float32
        )

    for item in range(len(dataset_val)):
        data_dict = dataset_val[item]
        data_dict = point_cloud_collate_fn([data_dict])
        data_dict = dict_to_device(data_dict, device)

        with torch.no_grad():
            out = model(data_dict)

        idx = data_dict["idx"].cpu().numpy()
        if out.shape[0] != len(idx):
            raise ValueError(
                f"Crop/index mismatch for {data_dict['file_name'][0]}: "
                f"model output has {out.shape[0]} points but idx has {len(idx)}. "
                "This usually means the dataset crop indices were not kept aligned during voxel downsampling."
            )
        file_name = data_dict["file_name"][0]
        predictions[file_name][idx] += out.softmax(dim=1).cpu().numpy()
        print(f"Done with crop {item + 1}/{len(dataset_val)}", end="\r")

    print()
    for key, data_dict in dataset_val.full_data_dicts.items():
        pred_max = np.argmax(predictions[key], axis=1)
        data_dict["predictions"] = pred_max

        file_name = key.split("/")[-1].split(".")[0]
        save_path = join(args.output_dir, f"{file_name}_predictions.las")
        print(f"Saving predictions: {save_path}")
        save_dict_to_laspy(data_dict, save_path)

    print("Inference complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference on point cloud data")
    args = parse_arguments(parser)
    main(args)