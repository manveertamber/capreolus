import torch


_hinge_loss = torch.nn.MarginRankingLoss(margin=1, reduction="mean")


def pair_softmax_loss(pos_neg_scores):
    scores = torch.stack(pos_neg_scores, dim=1)
    return torch.mean(1.0 - scores.softmax(dim=1)[:, 0])


def pair_hinge_loss(pos_neg_scores):
    label = torch.ones_like(pos_neg_scores[0])  # , dtype=torch.int)
    return _hinge_loss(pos_neg_scores[0], pos_neg_scores[1], label)


class SimilarityMatrix(torch.nn.Module):
    # based on SimmatModule from https://github.com/Georgetown-IR-Lab/cedr/blob/master/modeling_util.py
    # which is copyright (c) 2019 Georgetown Information Retrieval Lab, MIT license
    def __init__(self, padding=0):
        super().__init__()
        self.padding = padding

    # query_embed and doc_embed can be a list (eg for CEDR)
    def forward(self, query_embed, doc_embed, query_tok, doc_tok):
        simmat = []

        assert type(query_embed) == type(doc_embed)
        if not isinstance(query_embed, list):
            query_embed, doc_embed = [query_embed], [doc_embed]

        for a_emb, b_emb in zip(query_embed, doc_embed):
            BAT, A, B = a_emb.shape[0], a_emb.shape[1], b_emb.shape[1]

            if a_emb is None and b_emb is None:
                # exact match matrix
                sim = query_tok.reshape(BAT, A, 1).expand(BAT, A, B) == doc_tok.reshape(BAT, 1, B).expand(BAT, A, B).float()
            else:
                # cosine similarity matrix
                a_denom = a_emb.norm(p=2, dim=2).reshape(BAT, A, 1).expand(BAT, A, B) + 1e-9  # avoid 0div
                b_denom = b_emb.norm(p=2, dim=2).reshape(BAT, 1, B).expand(BAT, A, B) + 1e-9  # avoid 0div
                perm = b_emb.permute(0, 2, 1)
                sim = a_emb.bmm(perm) / (a_denom * b_denom)

            # set similarity values to 0 for <pad> tokens in query and doc (indicated by self.padding)
            nul = torch.zeros_like(sim)
            sim = torch.where(query_tok.reshape(BAT, A, 1).expand(BAT, A, B) == self.padding, nul, sim)
            sim = torch.where(doc_tok.reshape(BAT, 1, B).expand(BAT, A, B) == self.padding, nul, sim)

            simmat.append(sim)
        return torch.stack(simmat, dim=1)


class RbfKernel(torch.nn.Module):
    # based on KNRMRbfKernel from https://github.com/Georgetown-IR-Lab/cedr/blob/master/modeling_util.py
    # which is copyright (c) 2019 Georgetown Information Retrieval Lab, MIT license
    def __init__(self, initial_mu, initial_sigma, requires_grad=True):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.tensor(initial_mu), requires_grad=requires_grad)
        self.sigma = torch.nn.Parameter(torch.tensor(initial_sigma), requires_grad=requires_grad)

    def forward(self, data):
        adj = data - self.mu
        return torch.exp(-0.5 * adj * adj / self.sigma / self.sigma)


class RbfKernelBank(torch.nn.Module):
    # based on KNRMRbfKernelBank from https://github.com/Georgetown-IR-Lab/cedr/blob/master/modeling_util.py
    # which is copyright (c) 2019 Georgetown Information Retrieval Lab, MIT license
    def __init__(self, mus=None, sigmas=None, dim=1, requires_grad=True):
        super().__init__()
        self.dim = dim
        kernels = [RbfKernel(m, s, requires_grad=requires_grad) for m, s in zip(mus, sigmas)]
        self.kernels = torch.nn.ModuleList(kernels)

    def count(self):
        return len(self.kernels)

    def forward(self, data):
        return torch.stack([k(data) for k in self.kernels], dim=self.dim)


def create_emb_layer(weights, non_trainable=True):
    layer = torch.nn.Embedding(*weights.shape)
    layer.load_state_dict({"weight": torch.tensor(weights)})

    if non_trainable:
        layer.weight.requires_grad = False
    else:
        layer.weight.requires_grad = True

    return layer


def pool(tensor, dim=-1, pool_type="avg", ignore_padding=True):
    """
    conduct pool on given tensor over dimention dim

    :param tensor: torch.tensor, it should at least has dimension dim
    :param dim: int, which dimention to perform pool operation. default to be -1
    :param pool_type: str, 'max' / 'avg'
    :return:
    """
    supported_pool_type = ["max", "avg"]
    if pool_type not in supported_pool_type:
        raise ValueError(f"Unrecognized pooling type, should be one from {' '.join(supported_pool_type)}")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Unexpected input type: expected torch.Tensor yet got {type(tensor)}")
    if 0 <= dim and dim >= len(tensor.shape):
        raise ValueError(f"Attempt to perform operation on dim {dim} on tensor with shape {tensor.shape}, {len(tensor.shape)}")

    if pool_type == "max":
        # only rows whose sum is zeros are considered as padding position
        mask = (tensor.abs().sum(dim=-1) == 0).type(tensor.dtype)  # (B, T)
        tensor = (tensor - mask.unsqueeze(dim=-1) * 1e5) if ignore_padding else tensor
        max_values, indices = tensor.max(dim=dim)
        return max_values

    if pool_type == "avg":
        if not ignore_padding:
            return tensor.mean(dim=dim)  # (B, H)

        nonzero_pos = tensor.abs().sum(dim=-1) != 0  # (B, T)
        real_length = nonzero_pos.sum(dim=-1, keepdim=True)  # (B, T) -> (B, 1)

        tensor = tensor * nonzero_pos.unsqueeze(dim=-1)  # (B, T, H)
        # if (real_length == 0).sum() > 0:
        #     print(pool_type, real_length)
        return tensor.sum(dim=dim) / real_length  # since real_length are not supposed to be zero
