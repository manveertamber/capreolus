# based on https://github.com/Georgetown-IR-Lab/cedr/blob/master/modeling.py
# which is copyright (c) 2019 Georgetown Information Retrieval Lab, MIT license
import numpy as np
import torch
import torch.nn.functional as F

# from pytorch_pretrained_bert import BertModel
from pytorch_transformers import BertModel

from torch import nn

from capreolus.reranker.reranker import Reranker
from capreolus.extractor.berttext import BertText
from capreolus.reranker.common import create_emb_layer
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name

dtype = torch.FloatTensor

@Reranker.register
class CEDRKNRM(Reranker):
    EXTRACTORS = [BertText]

    @staticmethod
    def config():
        # gradacc = 8
        # batch = 2
        # lr = 0.001
        # bertlr = 0.00002
        # vanillaiters = 10
        jointbert = False
        freezebert = False
        return locals().copy()  # ignored by sacred

    def build(self):
        self.cfg = self.config.copy()
        self.model = CedrKNRM_class(self.cfg)

    def score(self, d):
        posd = {k[3:]: v for k, v in d.items() if k.startswith("pos")}
        negd = {k[3:]: v for k, v in d.items() if k.startswith("neg")}
        return [self.model(posd).view(-1), self.model(negd).view(-1)]

    def test(self, d):
        posd = {k[3:]: v for k, v in d.items() if k.startswith("pos")}
        return self.model(posd)

    def zero_grad(self, *args, **kwargs):
        self.model.zero_grad(*args, **kwargs)

    def get_optimizer(self):
        # TODO: support changing optimizor along the training?
        # the current version only allows separate the finetuning and training downstream task
        # in different experiment
        params = [(k, v) for k, v in self.model.named_parameters() if v.requires_grad]
        non_bert_params = {"params": [v for k, v in params if ".bert." not in k]}
        # we set a lower LR for the CustomBertModel (self.bert in BertRanker) only
        # params have names like: bert_ranker.bert.encoder.layer.9.attention.self.value.weight

        if not self.cfg["freezebert"]:
            bert_params = {"params": [v for k, v in params if ".bert." in k], "lr": self.cfg["bertlr"]}
            opt = torch.optim.Adam([bert_params, non_bert_params], lr=self.cfg["lr"])
        else:  # TODO: or just set bertlr == 0?
            # freeze bert parameters?
            opt = torch.optim.Adam([non_bert_params], lr=self.cfg["lr"])
        return opt


class BertRanker(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.BERT_MODEL = "bert-base-uncased"
        self.CHANNELS = 12 + 1  # from bert-base-uncased
        self.BERT_SIZE = 768  # from bert-base-uncased


class VanillaBertRanker(BertRanker):
    def __init__(self, config):
        super().__init__(config)
        self.bert = CustomBertModel.from_pretrained(
            self.BERT_MODEL,
            output_hidden_states=True,
        )
        self.dropout = torch.nn.Dropout(0.1)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)

    def encode(self, d):
        toks, mask, segs = d["toks"], d["mask"], d["segs"]  # (B, n_passages, L)
        segs = segs.long()

        qlen = self.cfg["maxqlen"]
        n_passages = toks.size(1)
        assert ( mask.size(1) == n_passages and segs.size(1) == n_passages)

        cls_results, query_results, doc_results = [], [], []
        for i in range(n_passages):
            results = self.bert(toks[:, i, :], segs[:, i, :], mask[:, i, :])
            cls_results.append(torch.stack([r[:, 0] for r in results], dim=0))  # (13, B, 768)
            query_results.append([r[:, 1:qlen+2] for r in results])
            doc_results.append(torch.stack([r[:, qlen+2:] for r in results], dim=0))  # each: (13, B, 512, 768)

        query_results = query_results[0]                            # q_vec of the last passage
        cls_results = torch.stack(cls_results, dim=-1).mean(dim=-1).unbind(dim=0)
        doc_results = torch.cat(doc_results, dim=2).unbind(dim=0)   # concatenate along the timestep dimension

        # result = self.bert(toks, segs.long(), mask)
        # QLEN = self.cfg["maxqlen"]
        #
        # doc_results = [r[:, QLEN + 2 : -1] for r in result]
        # query_results = [r[:, 1 : QLEN + 1] for r in result]
        # cls_results = [r[:, 0] for r in result]

        return cls_results, query_results, doc_results

    def forward(self, d):
        cls_reps, _, _ = self.encode(d)
        return self.cls(self.dropout(cls_reps[-1]))


# TODO inheriting BertRanker here may be messing up get_optimizer's LR logic
class CedrKNRM_class(BertRanker):
    def __init__(self, config):
        super().__init__(config)
        MUS = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        SIGMAS = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.001]
        self.bert_ranker = VanillaBertRanker(config)
        self.simmat = SimmatModule()
        self.kernels = KNRMRbfKernelBank(MUS, SIGMAS)

        ffw_dim = self.kernels.count() * self.CHANNELS + self.BERT_SIZE if self.cfg["jointbert"] else \
            self.kernels.count() * self.CHANNELS
        self.combine = torch.nn.Linear(ffw_dim, 1)
        self.epsilon = nn.Parameter(torch.tensor(1e-9), requires_grad=False)
        self.oniter = 0

    def is_finetuning(self):
        return self.oniter <= self.cfg["vanillaiters"]

    def forward(self, d):
        # if self.oniter <= self.cfg["vanillaiters"]:
        if self.is_finetuning():
            return self.bert_ranker(d)

        cls_reps, query_reps, doc_reps = self.bert_ranker.encode(d)
        simmat = self.simmat(query_reps, doc_reps, d["qmask"], d["dmask"])
        kernels = self.kernels(simmat)
        BATCH, KERNELS, VIEWS, QLEN, DLEN = kernels.shape
        kernels = kernels.reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        simmat = (
            simmat.reshape(BATCH, 1, VIEWS, QLEN, DLEN)
            .expand(BATCH, KERNELS, VIEWS, QLEN, DLEN)
            .reshape(BATCH, KERNELS * VIEWS, QLEN, DLEN)
        )
        result = kernels.sum(dim=3)  # sum over document
        mask = simmat.sum(dim=3) != 0.0  # which query terms are not padding?
        result = torch.where(mask, (result + self.epsilon).log(), mask.float())
        result = result.sum(dim=2)  # sum over query terms
        result = torch.cat([result, cls_reps[-1]], dim=1) if self.cfg["jointbert"] else result
        scores = self.combine(result)  # linear combination over kernels
        return scores


class SimmatModule(torch.nn.Module):
    def __init__(self, padding=0):
        super().__init__()
        self.padding = nn.Parameter(torch.tensor(padding), requires_grad=False)
        self.epsilon = nn.Parameter(torch.tensor(1e-9), requires_grad=False)
        self._hamming_index_loaded = None
        self._hamming_index = None

    def forward(self, query_embed, doc_embed, query_tok, doc_tok):
        simmat = []

        for a_emb, b_emb in zip(query_embed, doc_embed):
            BAT, A, B = a_emb.shape[0], a_emb.shape[1], b_emb.shape[1]
            # embeddings -- cosine similarity matrix
            a_denom = a_emb.norm(p=2, dim=2).reshape(BAT, A, 1).expand(BAT, A, B) + self.epsilon  # avoid 0div
            b_denom = b_emb.norm(p=2, dim=2).reshape(BAT, 1, B).expand(BAT, A, B) + self.epsilon  # avoid 0div
            perm = b_emb.permute(0, 2, 1)
            sim = a_emb.bmm(perm)
            sim = sim / (a_denom * b_denom)

            # nullify padding
            nul = torch.zeros_like(sim)
            sim = torch.where(query_tok.reshape(BAT, A, 1).expand(BAT, A, B) == self.padding, nul, sim)
            sim = torch.where(doc_tok.reshape(BAT, 1, B).expand(BAT, A, B) == self.padding, nul, sim)

            simmat.append(sim)
        return torch.stack(simmat, dim=1)


class KNRMRbfKernelBank(torch.nn.Module):
    def __init__(self, mus=None, sigmas=None, dim=1, requires_grad=True):
        super().__init__()
        self.dim = dim
        kernels = [KNRMRbfKernel(m, s, requires_grad=requires_grad) for m, s in zip(mus, sigmas)]
        self.kernels = torch.nn.ModuleList(kernels)

    def count(self):
        return len(self.kernels)

    def forward(self, data):
        return torch.stack([k(data) for k in self.kernels], dim=self.dim)


class KNRMRbfKernel(torch.nn.Module):
    def __init__(self, initial_mu, initial_sigma, requires_grad=True):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.tensor(initial_mu), requires_grad=requires_grad)
        self.sigma = torch.nn.Parameter(torch.tensor(initial_sigma), requires_grad=requires_grad)

    def forward(self, data):
        adj = data - self.mu
        return torch.exp(-0.5 * adj * adj / self.sigma / self.sigma)


class CustomBertModel(BertModel):
    """
    Based on pytorch_pretrained_bert.BertModel, but also outputs un-contextualized embeddings.
    """

    def forward(self, input_ids, token_type_ids, attention_mask):
        """ Based on pytorch_pretrained_bert.BertModel """
        embedding_output = self.embeddings(input_ids, token_type_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        last_hidden_layer, encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=[None] * 12)  # * self.config.num_hidden_layers)

        return (embedding_output, ) + encoded_layers