import importlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ALL_MODEL_DEPS = [
    "torch",
    "torch_geometric",
    "torch_scatter",
    "einops",
    "timm",
    "pointops",
    "spconv.pytorch",
    "addict",
    "flash_attn",
    "lib.pointrope",
]


def _require_modules(modules):
    missing = []
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as exc:
            missing.append(f"{module} ({exc.__class__.__name__})")
    if missing:
        pytest.skip("Missing optional dependencies: " + ", ".join(missing))


def _load_config(name):
    return str(ROOT / "config" / name)


def _load_model(name, config_name):
    _require_modules(ALL_MODEL_DEPS)
    try:
        from models.model_loader import get_model
    except Exception as exc:
        pytest.skip(f"Unable to import model_loader: {exc}")
    return get_model(name, _load_config(config_name), verbose=False)


def test_init_point_transformer_v2():
    model = _load_model("point_transformer_v2", "ptv2_example.yaml")
    assert model is not None


def test_init_point_transformer_v3():
    model = _load_model("point_transformer_v3", "ptv3_example.yaml")
    assert model is not None


def test_init_litept():
    model = _load_model("litept", "litept_example.yaml")
    assert model is not None


def test_init_spconv_unet():
    model = _load_model("spconv_unet", "spconv_example.yaml")
    assert model is not None


def test_init_oacnns():
    model = _load_model("oacnns", "oacnns_example.yaml")
    assert model is not None
