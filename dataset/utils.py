import torch
import numpy as np
import spconv.pytorch as spconv

############################################################################
# dict stuff


def dict_from_idx(in_dict: dict, idx: int) -> dict:
    """
    Get new new dictronary with the same keys as in_dict but only the values at index idx.
    DOES NOT HANDLE OFFSETS OR STUFF LIKE THAT! Only for simple dicts with values of shape (N,).
    """
    out_dict = {}
    for key, value in in_dict.items():
        out_dict[key] = value[idx]
    return out_dict


def dict_to_device(data_dict: dict, device: torch.device) -> dict:
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            data_dict[k] = v.to(device)
        else:
            data_dict[k] = v
    return data_dict


def dict_to_numpy(data_dict: dict) -> dict:
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            data_dict[k] = v.cpu().numpy()
        else:
            data_dict[k] = v
    return data_dict


def dict_to_torch(data_dict: dict, device: torch.device) -> dict:
    for k, v in data_dict.items():
        if isinstance(v, np.ndarray):
            data_dict[k] = torch.from_numpy(v).to(device)
        else:
            data_dict[k] = v
    return data_dict


def dict_to_spconv(data_dict: dict) -> spconv.SparseConvTensor:
    discrete_coord = data_dict["disc_coords"]
    feat = data_dict["feats"]
    offset = data_dict["offsets"]
    batch = offset2batch(offset)

    x = spconv.SparseConvTensor(
        features=feat,
        indices=torch.cat([batch.unsqueeze(-1), discrete_coord], dim=1)
        .int()
        .contiguous(),
        spatial_shape=torch.add(torch.max(discrete_coord, dim=0).values, 96).tolist(),
        batch_size=batch[-1].tolist() + 1,
    )
    return x


def dict_rename_for_pointcept(data_dict):
    renamed_dict = {}
    for key, value in data_dict.items():
        if key == "coords":
            renamed_dict["coord"] = value
        elif key == "feats":
            renamed_dict["feat"] = value
        elif key == "disc_coords":
            renamed_dict["grid_coord"] = value
        elif key == "offsets":
            renamed_dict["offset"] = value
        else:
            renamed_dict[key] = value
    return renamed_dict


############################################################################
# offset stuff


@torch.inference_mode()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.inference_mode()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


############################################################################
# other stuff...


@torch.inference_mode()
def get_attention_entropy(q, k, scale):
    attn_scores = torch.einsum("bhnd,bhmd->bhnm", q.float(), k.float()) * scale
    attn_probs = attn_scores.softmax(dim=-1)
    entropy = -(attn_probs * (attn_probs + 1e-9).log()).sum(-1).mean()
    print(f"Attention entropy: {entropy.item():.4f}")
    return entropy
