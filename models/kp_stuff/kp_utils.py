import torch


def index_select(inputs: torch.Tensor, indices: torch.Tensor, dim: int) -> torch.Tensor:
    """Advanced indices select.

    Returns a tensor `output` which indexes the `inputs` tensor along dimension `dim` using the entries in `indices`
    which is a `LongTensor`.

    Different from `torch.indices_select`, `indices` does not have to be 1-D. The `dim`-th dimension of `inputs` will
    be expanded to the number of dimensions in `indices`.

    For example, suppose the shape `inputs` is $(a_0, a_1, ..., a_{n-1})$, the shape of `indices` is
    $(b_0, b_1, ..., b_{m-1})$, and `dim` is $i$, then `output` is $(n+m-1)$-d tensor, whose shape is
    $(a_0, ..., a_{i-1}, b_0, b_1, ..., b_{m-1}, a_{i+1}, ..., a_{n-1})$.

    Args:
        inputs (Tensor): (a_0, a_1, ..., a_{n-1})
        indices (LongTensor): (b_0, b_1, ..., b_{m-1})
        dim (int): The dimension to index.

    Returns:
        output (Tensor): (a_0, ..., a_{dim-1}, b_0, ..., b_{m-1}, a_{dim+1}, ..., a_{n-1})
    """
    outputs = inputs.index_select(dim, indices.view(-1))

    if indices.dim() > 1:
        if dim < 0:
            dim += inputs.dim()
        output_shape = inputs.shape[:dim] + indices.shape + inputs.shape[dim + 1 :]
        outputs = outputs.view(*output_shape)

    return outputs


def gather(x, idx, method=2):
    """
    implementation of a custom gather operation for faster backwards.
    :param x: input with shape [N, D_1, ... D_d]
    :param idx: indexing with shape [n_1, ..., n_m]
    :param method: Choice of the method
    :return: x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    """

    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        # for i, ni in enumerate(idx.size()[1:]):
        #     x = x.unsqueeze(i + 1)
        #     new_s = list(x.size())
        #     new_s[i + 1] = ni
        #     x = x.expand(new_s)
        # n = len(idx.size())
        # for i, di in enumerate(x.size()[n:]):
        #     idx = idx.unsqueeze(i + n)
        #     new_s = list(idx.size())
        #     new_s[i + n] = di
        #     idx = idx.expand(new_s)
        # return x.gather(0, idx)
        # Use index_select to avoid expanding idx to feature dims (lower peak memory).
        flat_idx = idx.view(-1)
        out = x.index_select(0, flat_idx)
        if idx.dim() > 1:
            out = out.view(*idx.shape, *x.shape[1:])
        return out
    else:
        raise ValueError("Unkown method")


def radius_gaussian(sq_r, sig, eps=1e-9):
    """
    Compute a radius gaussian (gaussian of distance)
    :param sq_r: input radiuses [dn, ..., d1, d0]
    :param sig: extents of gaussians [d1, d0] or [d0] or float
    :return: gaussian of sq_r [dn, ..., d1, d0]
    """
    return torch.exp(-sq_r / (2 * sig**2 + eps))
