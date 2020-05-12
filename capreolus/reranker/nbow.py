import torch
import torch.nn as nn
import torch.nn.functional as F

from capreolus.reranker import Reranker
from capreolus.reranker.common import create_emb_layer, pool
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


class Nbow_class(nn.Module):
    def __init__(self, extractor, config):
        super(Nbow_class, self).__init__()
        self.cfg = config
        self.embedding = create_emb_layer(extractor.embeddings, non_trainable=False)

    def forward(self, documents, queries, query_ids):
        # print(f"doc: {documents.shape}; query: {queries.shape}")
        pool_type = self.cfg["pool_type"]
        documents, queries = self.embedding(documents), self.embedding(queries)
        documents = pool(documents, dim=1, pool_type=pool_type)   # (B, H)
        queries = pool(queries, dim=1, pool_type=pool_type)  # (B, H)

        # d_norm, q_norm = documents.norm(dim=-1, keepdim=True), queries.norm(dim=-1, keepdim=True)
        # cos = torch.div(documents * queries, (q_norm * d_norm) + 1e-5).sum(dim=-1)  # (B, )
        cos = F.cosine_similarity(queries, documents, dim=-1)  # (B, )
        return cos


class NBOW(Reranker):
    name = "NBOW"
    decription = ""

    @staticmethod
    def config():
        pool_type = "max"

    def build(self):
        if not hasattr(self, "model"):
            self.model = Nbow_class(self["extractor"], self.cfg)

        return self.model

    def score(self, d):
        query_idf = d["query_idf"]
        query_sentence = d["query"]
        pos_sentence, neg_sentence = d["posdoc"], d["negdoc"]
        return [
            self.model(pos_sentence, query_sentence, query_idf).view(-1),
            self.model(neg_sentence, query_sentence, query_idf).view(-1),
        ]

    def test(self, d):
        query_idf = d["query_idf"]
        query_sentence = d["query"]
        pos_sentence = d["posdoc"]

        return self.model(pos_sentence, query_sentence, query_idf).view(-1)
