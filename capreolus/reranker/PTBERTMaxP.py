import torch
from torch import nn
from transformers import AutoModelForSequenceClassification

from capreolus import ConfigOption, Dependency
from capreolus.reranker import Reranker


class ElectraRelevanceHead(nn.Module):
    """ BERT-style ClassificationHead (i.e., out_proj only -- no dense). See transformers.TFElectraClassificationHead """

    def __init__(self, dropout, out_proj, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = dropout
        self.out_proj = out_proj

    def call(self, inputs, **kwargs):
        x = inputs[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class PTBERTMaxP_Class(nn.Module):
    def __init__(self, extractor, config, *args, **kwargs):
        super(PTBERTMaxP_Class, self).__init__(*args, **kwargs)
        self.extractor = extractor

        if config["pretrained"] == "electra-base-msmarco":
            self.bert = AutoModelForSequenceClassification.from_pretrained("Capreolus/electra-base-msmarco")
            dropout, fc = self.bert.classifier.dropout, self.bert.classifier.out_proj
            self.bert.classifier = ElectraRelevanceHead(dropout, fc)
        elif config["pretrained"] == "bert-base-msmarco":
            self.bert = AutoModelForSequenceClassification.from_pretrained("Capreolus/bert-base-msmarco")
        else:
            self.bert = AutoModelForSequenceClassification.from_pretrained(
                config["pretrained"], hidden_dropout_prob=config["hidden_dropout_prob"]
            )

        self.config = config

    def forward(self,  doc_bert_input, doc_mask, doc_seg):
        """ Returns logits of shape [2] """
        # doc_bert_input, doc_mask, doc_seg = x[0], x[1], x[2]
        batch_size = doc_bert_input.size(0)
        maxseqlen = self.extractor.config["maxseqlen"]

        if len(doc_bert_input.shape) == 3:  # training time
            doc_bert_input = torch.reshape(doc_bert_input[:, 0, :], [batch_size, maxseqlen])
            doc_mask = torch.reshape(doc_mask[:, 0, :], [batch_size, maxseqlen])
            doc_seg = torch.reshape(doc_seg[:, 0, :], [batch_size, maxseqlen])

        # doc_bert_input = torch.reshape(doc_bert_input, [batch_size, maxseqlen])
        # doc_mask = torch.reshape(doc_mask, [batch_size, maxseqlen])
        # doc_seg = torch.reshape(doc_seg, [batch_size, maxseqlen])

        if "roberta" in self.config["pretrained"]:
            doc_seg = torch.zeros_like(doc_mask)  # since roberta does not have segment input
        passage_scores = self.bert(doc_bert_input, attention_mask=doc_mask, token_type_ids=doc_seg)[0]
        return passage_scores

    def predict_step(self, posdoc_bert_input, posdoc_mask, posdoc_seg):
        """
        Scores each passage and applies max pooling over it.
        """
        # posdoc_bert_input, posdoc_mask, posdoc_seg, negdoc_bert_input, negdoc_mask, negdoc_seg = data
        batch_size = posdoc_bert_input.size(0)
        num_passages = self.extractor.config["numpassages"]
        maxseqlen = self.extractor.config["maxseqlen"]

        passage_position = torch.sum(posdoc_mask * posdoc_seg, dim=-1)  # (B, P)
        passage_mask = (passage_position > 5).double()  # (B, P)

        # import pdb
        # pdb.set_trace()
        posdoc_bert_input = torch.reshape(posdoc_bert_input, [batch_size * num_passages, maxseqlen])
        posdoc_mask = torch.reshape(posdoc_mask, [batch_size * num_passages, maxseqlen])
        posdoc_seg = torch.reshape(posdoc_seg, [batch_size * num_passages, maxseqlen])

        passage_scores = self(posdoc_bert_input, posdoc_mask, posdoc_seg)[:, 1]
        import pdb
        pdb.set_trace()
        passage_scores = torch.reshape(passage_scores, [batch_size, num_passages])

        if self.config["aggregation"] == "max":
            passage_scores = torch.max(passage_scores, dim=1).values
        elif self.config["aggregation"] == "first":
            passage_scores = passage_scores[:, 0]
        elif self.config["aggregation"] == "sum":
            passage_scores = torch.sum(passage_mask * passage_scores, dim=1)
        elif self.config["aggregation"] == "avg":
            passage_scores = torch.sum(passage_mask * passage_scores, dim=1) / torch.sum(passage_mask)
        else:
            raise ValueError("Unknown aggregation method: {}".format(self.config["aggregation"]))

        return passage_scores


@Reranker.register
class PTBERTMaxP(Reranker):
    """
    TensorFlow implementation of BERT-MaxP.

    Deeper Text Understanding for IR with Contextual Neural Language Modeling. Zhuyun Dai and Jamie Callan. SIGIR 2019.
    https://arxiv.org/pdf/1905.09217.pdf
    """

    module_name = "PTBERTMaxP"

    dependencies = [
        Dependency(key="extractor", module="extractor", name="bertpassage"),
        Dependency(key="trainer", module="trainer", name="pytorch"),
    ]
    config_spec = [
        ConfigOption(
            "pretrained",
            "bert-base-uncased",
            "Pretrained model: bert-base-uncased, bert-base-msmarco, electra-base-msmarco, or HuggingFace supported models",
        ),
        ConfigOption("aggregation", "max"),
        ConfigOption("hidden_dropout_prob", 0.1, "The dropout probability of BERT-like model's hidden layers."),
    ]

    def build_model(self):
        self.model = PTBERTMaxP_Class(self.extractor, self.config)
        return self.model

    def score(self, d):
        return [
            self.model(d["pos_bert_input"], d["pos_seg"], d["pos_mask"])[:, 1],
            self.model(d["neg_bert_input"], d["neg_seg"], d["neg_mask"])[:, 1],
        ]

    def test(self, d):
        return self.model.predict_step(d["pos_bert_input"], d["pos_seg"], d["pos_mask"]).view(-1)


