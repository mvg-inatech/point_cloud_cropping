import itertools
import random
from copy import deepcopy

from dataset.sub_cloud_calc import calculate_sub_clouds_grid, calculate_sub_clouds
from dataset.base_dataset import BaseDataset
from dataset.utils import dict_from_idx

from common.filter import get_sub_idx_function


class LambdaRange(object):
    """Class to handle lambda range for sub cloud calculation. Can be radius, std or exponent."""

    def __init__(self, base_lambda, lambda_range, use_lambda_range):
        self.base_lambda = base_lambda
        self.lambda_range = lambda_range
        self.use_lambda_range = use_lambda_range

    def get_lambda(self):
        if self.use_lambda_range:
            return random.uniform(
                self.base_lambda - self.lambda_range[0],
                self.base_lambda + self.lambda_range[1],
            )
        else:
            return self.base_lambda


class LargeScaleDataset(BaseDataset):
    """
    Dataset class for large scale point clouds (whole large scenes like pl3d or SemanticBridge).
    It calculates sub clouds based on a 3D grid overlay and stores them in self.sub_clouds.
    Requires much memory but allows for faster data loading during training.
    """

    def __init__(self, dataset_config, split):
        super().__init__(dataset_config, split)
        self.lambda_p = dataset_config.lambda_p
        self.grid_overlay = (
            dataset_config.grid_overlay
            if hasattr(dataset_config, "grid_overlay")
            else None
        )
        self.min_pts = dataset_config.min_pts
        self.get_sub_idx = get_sub_idx_function(dataset_config.sub_cloud_method)

        self.sub_clouds = []
        self.full_data_dicts = {}

        self.init_sub_clouds()

    def init_single_cloud(self, idx):
        file_name = self.file_paths[idx]
        data_dict = self.read_and_preprocess(idx)
        self.full_data_dicts[file_name] = data_dict
        if self.grid_overlay is not None:
            self.sub_clouds.append(
                calculate_sub_clouds_grid(
                    data_dict,
                    file_name,
                    self.lambda_p,
                    self.grid_overlay,
                    self.get_sub_idx,
                    self.min_pts,
                )
            )
        else:
            self.sub_clouds.append(
                calculate_sub_clouds(
                    data_dict,
                    file_name,
                    self.lambda_p,
                    self.get_sub_idx,
                    self.min_pts,
                )
            )

    def clear_sub_clouds(self):
        self.sub_clouds = []
        self.full_data_dicts = {}

    def chain_sub_clouds(self):
        self.sub_clouds = list(itertools.chain.from_iterable(self.sub_clouds))

    def __len__(self):
        return len(self.sub_clouds) * self.loops

    def init_sub_clouds(self):
        self.clear_sub_clouds()
        for idx in range(len(self.file_paths)):
            self.init_single_cloud(idx)
            print(
                "Done with file {} -> splitted into {} crops".format(
                    self.file_paths[idx], len(self.sub_clouds[-1])
                )
            )
        self.chain_sub_clouds()
        if len(self.sub_clouds) == 0:
            raise ValueError(
                "No point clouds loaded -> most probably wrong directory or wrong ending{}".format(
                    self.file_ending
                )
            )

    def __getitem__(self, idx):
        idx_to_use = idx % len(self.sub_clouds)

        sub_cloud = self.sub_clouds[idx_to_use]
        data_dict = deepcopy(
            dict_from_idx(
                self.full_data_dicts[sub_cloud.file_name],
                sub_cloud.idx,
            )
        )

        data_dict["coords"] -= sub_cloud.center
        data_dict["probabilities"] = sub_cloud.probabilities
        data_dict = self.normalize_dict(data_dict)
        if self.split == "train":
            data_dict = self.transform(data_dict)
        else:
            data_dict["idx"] = sub_cloud.idx
            data_dict["file_name"] = [sub_cloud.file_name]
        data_dict = self.discretize_coords(data_dict)
        data_dict = self.create_features(data_dict)
        data_dict["pos"] = sub_cloud.center.reshape(1, 3)
        return data_dict
