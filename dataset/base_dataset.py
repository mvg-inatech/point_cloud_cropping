import random
from pathlib import Path
import torch
import numpy as np

from dataset.load_dataset_cloud import get_pc_loader
from common.voxelize import voxelize
from dataset.augmentation import (
    RandomPointJitter,
    RandomPointDrop,
    RandomPointScale,
    RandomPointRotateZ,
    RandomPointFlip,
    GaussianPointNoise,
    GaussianColorNoise,
    ChromaticColorTranslation,
    RandomColorDrop,
)
from dataset.utils import dict_from_idx, dict_to_torch


############################################################################
# parsing stuff


def parse_dir_for_x_file(
    directory: str,
    x_ending: str,
    recursive: bool = False,
) -> list:
    path_obj = Path(directory)
    pattern = f"*{x_ending}"
    if recursive:
        files = list(path_obj.rglob(pattern))
    else:
        files = list(path_obj.glob(pattern))

    wanted_files = [str(f) for f in files if f.is_file()]
    print(f"Found {len(wanted_files)} {x_ending} files in dir: {directory}")
    return wanted_files


############################################################################
# BaseDataset class


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_config, split):
        self.voxel_size = dataset_config.voxel_size
        self.loops = dataset_config.loops
        self.file_ending = dataset_config.ending
        self.feat_list = dataset_config.feat_list
        if split == "train":
            root_dir = dataset_config.train_dir
        elif split == "val":
            root_dir = dataset_config.val_dir
        elif split == "test":
            root_dir = dataset_config.test_dir
        else:
            raise ValueError(f"Invalid split: {split}")
        self.split = split
        self.file_paths = parse_dir_for_x_file(root_dir, self.file_ending)

        self.pc_loader = get_pc_loader(dataset_config.class_name)

    def __len__(self):
        return len(self.file_paths) * self.loops

    def read_and_preprocess(self, idx):
        data_dict = self.pc_loader(self.file_paths[idx])
        idx, _ = voxelize(data_dict["coords"], self.voxel_size)
        data_dict = dict_from_idx(data_dict, idx)
        return data_dict

    def discretize_coords(self, data_dict):
        data_dict["disc_coords"] = np.floor(
            (data_dict["coords"] - np.min(data_dict["coords"], axis=0))
            / np.array(self.voxel_size)
        ).astype(np.int32)
        return data_dict

    def normalize_dict(self, data_dict):
        if "colors" in data_dict:
            data_dict["colors"] = data_dict["colors"] / 255.0
        if "intensity" in data_dict:
            data_dict["intensity"] = data_dict["intensity"] / 255.0
        return data_dict

    def transform(self, data_dict):
        data_dict = RandomPointDrop()(data_dict)
        if random.random() > 0.5:
            data_dict = RandomPointJitter()(data_dict)
        if random.random() > 0.5:
            data_dict = RandomPointScale()(data_dict)
        if random.random() > 0.5:
            data_dict = RandomPointRotateZ()(data_dict)
        if random.random() > 0.5:
            data_dict = RandomPointFlip()(data_dict)
        if random.random() > 0.5:
            data_dict = GaussianPointNoise()(data_dict)
        # feats
        if random.random() > 0.5:
            data_dict = GaussianColorNoise()(data_dict)
        if random.random() > 0.5:
            data_dict = ChromaticColorTranslation()(data_dict)
        if random.random() > 0.5:
            data_dict = RandomColorDrop()(data_dict)
        return data_dict

    def create_features(self, data_dict):
        """
        Create final feature vector for model.
        """
        for feat in self.feat_list:
            if feat not in data_dict.keys():
                raise ValueError(
                    f"Feature {feat} not found in data dict - wrong feat_list?"
                )
        features = []
        for feat in self.feat_list:
            sub_feat = data_dict[feat]
            if len(sub_feat.shape) == 1:
                sub_feat = sub_feat.reshape(-1, 1)
            features.append(sub_feat)
        data_dict["feats"] = np.concatenate(features, axis=1).astype(np.float32)
        return data_dict


############################################################################
# point cloud dict collate function


def point_cloud_collate_fn(pre_batch):
    collated_batch = {}
    offsets = []
    for i, batch in enumerate(pre_batch):
        for key, value in batch.items():
            if key not in collated_batch:
                collated_batch[key] = []
            if isinstance(value, np.ndarray):
                collated_batch[key].append(value)
            elif isinstance(value, list):
                collated_batch[key].extend(value)
            else:
                continue  # Do not care about non array values for now
        offsets.append(
            batch["coords"].shape[0]
            if i == 0
            else offsets[-1] + batch["coords"].shape[0]
        )

    for key, value in collated_batch.items():
        if isinstance(value[0], np.ndarray):
            collated_batch[key] = np.concatenate(value, axis=0)

    collated_batch["offsets"] = np.asarray(offsets)
    collated_batch = dict_to_torch(collated_batch, device=torch.device("cpu"))
    return collated_batch
