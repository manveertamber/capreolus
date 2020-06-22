import sys
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from transformers import TFBertForSequenceClassification

from profane import ConfigOption, Dependency
from capreolus.reranker.base import Reranker
from capreolus.utils.loginit import get_logger


class TFBERTMaxP_Class(tf.keras.layers.Layer):
    def __init__(self, extractor, config, *args, **kwargs):
        super(TFBERTMaxP_Class, self).__init__(*args, **kwargs)
        self.extractor = extractor
        self.bert = TFBertForSequenceClassification.from_pretrained(config["pretrained"], hidden_dropout_prob=0.1)
        self.config = config

    def call(self, x, **kwargs):
        doc_bert_input, doc_mask, doc_seg = x[0], x[1], x[2]

        batch_size = tf.shape(doc_bert_input)[0]
        num_passages = self.extractor.config["numpassages"]
        maxseqlen = self.extractor.config["maxseqlen"]

        doc_bert_input = tf.reshape(doc_bert_input, [batch_size * num_passages, maxseqlen])
        doc_mask = tf.reshape(doc_mask, [batch_size * num_passages, maxseqlen])
        doc_seg = tf.reshape(doc_seg, [batch_size * num_passages, maxseqlen])

        passage_scores = self.bert(doc_bert_input, attention_mask=doc_mask, token_type_ids=doc_seg)[0][:, 0]
        passage_scores = tf.reshape(passage_scores, [batch_size, num_passages])

        return passage_scores

    def predict_step(self, data):
        data = data_adapter.expand_1d(data)
        x, _, _ = data_adapter.unpack_x_y_sample_weight(data)
        passage_scores = self.score(x, training=False)

        return tf.math.reduce_max(passage_scores, axis=1)

    def score(self, x, **kwargs):
        posdoc_bert_input, posdoc_mask, posdoc_seg, negdoc_bert_input, negdoc_mask, negdoc_seg = x

        return self.call((posdoc_bert_input, posdoc_mask, posdoc_seg))

    def score_pair(self, x, **kwargs):
        posdoc_bert_input, posdoc_mask, posdoc_seg, negdoc_bert_input, negdoc_mask, negdoc_seg = x

        pos_score = self.call((posdoc_bert_input, posdoc_mask, posdoc_seg))
        neg_score = self.call((negdoc_bert_input, negdoc_mask, negdoc_seg))
        batch_size = tf.shape(pos_score)[0]

        stacked_score = tf.stack([pos_score, neg_score], axis=2)
        return stacked_score

    # def call(self, x, **kwargs):
    #     pos_toks, posdoc_mask, neg_toks, negdoc_mask, query_toks, query_mask = x[0], x[1], x[2], x[3], x[4], x[5]
    #     batch_size = tf.shape(pos_toks)[0]
    #     doclen = self.extractor.cfg["maxdoclen"]
    #     qlen = self.extractor.cfg["maxqlen"]
    #
    #     cls = tf.cast(tf.fill([batch_size, 1], self.clsidx, name="clstoken"), tf.int64)
    #     sep_1 = tf.cast(tf.fill([batch_size, 1], self.sepidx, name="septoken1"), tf.int64)
    #     sep_2 = tf.cast(tf.fill([batch_size, 1], self.sepidx, name="septoken2"), tf.int64)
    #     ones = tf.ones([batch_size, 1], dtype=tf.int64)
    #
    #     passagelen = self.config["passagelen"]
    #     stride = self.config["stride"]
    #     # TODO: Integer division would mean that we round down - the last passage would be lost
    #     num_passages = (doclen - passagelen) // stride
    #     # The passage level scores will be stored in these arrays
    #     pos_passage_scores = tf.TensorArray(tf.float32, size=doclen // passagelen, dynamic_size=False)
    #     neg_passage_scores = tf.TensorArray(tf.float32, size=doclen // passagelen, dynamic_size=False)
    #
    #     i = 0
    #     idx = 0
    #
    #     while idx < num_passages:
    #         # Get a passage and the corresponding mask
    #         pos_passage = pos_toks[:, i : i + passagelen]
    #         pos_passage_mask = posdoc_mask[:, i : i + passagelen]
    #         neg_passage = neg_toks[:, i : i + passagelen]
    #         neg_passage_mask = negdoc_mask[:, i : i + passagelen]
    #
    #         # Prepare the input to bert
    #         query_pos_passage_tokens_tensor = tf.concat([cls, query_toks, sep_1, pos_passage, sep_2], axis=1)
    #         query_pos_passage_mask = tf.concat([ones, query_mask, ones, pos_passage_mask, ones], axis=1)
    #         query_neg_passage_tokens_tensor = tf.concat([cls, query_toks, sep_1, neg_passage, sep_2], axis=1)
    #         query_neg_passage_mask = tf.concat([ones, query_mask, ones, neg_passage_mask, ones], axis=1)
    #         query_passage_segments_tensor = tf.concat(
    #             [tf.zeros([batch_size, qlen + 2]), tf.ones([batch_size, passagelen + 1])], axis=1
    #         )
    #
    #         # Actual bert scoring
    #         pos_passage_score = self.bert(
    #             query_pos_passage_tokens_tensor,
    #             attention_mask=query_pos_passage_mask,
    #             token_type_ids=query_passage_segments_tensor,
    #         )[0][:, 0]
    #         neg_passage_score = self.bert(
    #             query_neg_passage_tokens_tensor,
    #             attention_mask=query_neg_passage_mask,
    #             token_type_ids=query_passage_segments_tensor,
    #         )[0][:, 0]
    #         pos_passage_scores = pos_passage_scores.write(idx, pos_passage_score)
    #         neg_passage_scores = neg_passage_scores.write(idx, neg_passage_score)
    #
    #         idx += 1
    #         i += stride
    #
    #     posdoc_scores = tf.math.reduce_max(pos_passage_scores.stack(), axis=0)
    #     negdoc_scores = tf.math.reduce_max(neg_passage_scores.stack(), axis=0)
    #     return tf.stack([posdoc_scores, negdoc_scores], axis=1)


@Reranker.register
class TFBERTMaxP(Reranker):
    module_name = "TFBERTMaxP"

    dependencies = [
        Dependency(key="extractor", module="extractor", name="bertpassage"),
        Dependency(key="trainer", module="trainer", name="tensorflow"),
    ]
    config_spec = [
        ConfigOption("pretrained", "bert-base-uncased", "Hugging face transformer pretrained model"),
        ConfigOption("passagelen", 100, "Passage length"),
        ConfigOption("dropout", 0.1, "Dropout for the linear layers in BERT"),
        ConfigOption("stride", 20, "Stride")
    ]

    def build_model(self):
        self.model = TFBERTMaxP_Class(self.extractor, self.config)
        return self.model
