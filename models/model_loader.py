from models.point_transformer_v2 import PointTransformerV2
from models.point_transformer_v3 import PointTransformerV3
from models.litept import LitePT
from models.sonata import Sonata
from models.spconv_unet import SpConvUNet
from models.oacnns import OACNNs
from common.parser import yaml_cfg_to_class
from prettytable import PrettyTable


def count_parameters(model, verbose=True):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    trainable_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            total_params += parameter.numel()
            continue
        param = parameter.numel()
        table.add_row([name, param])
        trainable_params += param
        total_params += param
    if verbose:
        print(table)
    print(f"Total Params: {total_params} -- Trainable Params: {trainable_params}")
    return trainable_params


def get_model(name, config_dir, verbose=False):
    config = yaml_cfg_to_class(config_dir, "name", "model_config")
    if name == "point_transformer_v2":
        model = PointTransformerV2(config)
    elif name == "spconv_unet":
        model = SpConvUNet(config)
    elif name == "oacnns":
        model = OACNNs(config)
    elif name == "point_transformer_v3":
        model = PointTransformerV3(config)
    elif name == "sonata":
        # slightly modified PointTransformerV3
        model = Sonata(config)
    elif name == "litept":
        model = LitePT(config)
    else:
        raise NotImplementedError
    print("Model: {} ".format(name))
    count_parameters(model, verbose=verbose)
    return model
