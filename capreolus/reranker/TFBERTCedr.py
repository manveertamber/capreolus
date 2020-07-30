import numpy as np
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from transformers import TFBertModel

from profane import ConfigOption, Dependency
from capreolus.reranker import Reranker


class RBFKernel(tf.keras.layers.Layer):
    def __init__(self, init_mu, init_sigma, postfix=0, trainable=True, *args, **kwargs):
        super(RBFKernel, self).__init__(*args, **kwargs)
        self.mu = tf.Variable(initial_value=init_mu, trainable=trainable, name=f"mu-{postfix}")
        self.sigma = tf.Variable(initial_value=init_sigma, trainable=trainable, name=f"sigma-{postfix}")

    def call(self, x, **kwargs):
        adj = x - self.mu
        return tf.math.exp(-0.5 * (adj * adj) / (self.sigma * self.sigma + 1e-6))


class CEDRKNRM(tf.keras.layers.Layer):
    """
    A KNRM model starting from simmat matrix, used as an affix of BERT
    """
    def __init__(self, mus=None, sigmas=None, train_kernels=True, *args, **kwargs):
        """
        :param query_reps: (B, n_layers, Q, n_hidden)
        :param query_mask: (B, Q), 1 for non-pad entries and 0 for pad entries
        :param doc_reps: (B, n_layers, D, n_hidden)
        :param doc_mask: (B, D), 1 for non-pad entries and 0 for pad entries
        :param mus: a list of means for kernal
        :param sigmas: a list of variance for kernal
        mus and sigmas should have same size
        """
        super(CEDRKNRM, self).__init__(*args, **kwargs)
        if not mus:
            mus = np.concatenate([np.linspace(-0.9, 0.9, 10), np.array([1.])]).astype(np.float32)
        if not sigmas:
            sigmas = np.array([0.1] * 10 + [0.001], dtype=np.float32)

        self.kernels = [
            RBFKernel(init_mu=mu, init_sigma=sigma, trainable=train_kernels, postfix=i)
            for i, (mu, sigma) in enumerate(zip(mus, sigmas))]

    def call(self, x, **kwargs):
        """ mask: 1 represents valid positions, 0 represents padded positions """
        query_reps, doc_reps, query_mask, doc_mask = x

        query_mask, doc_mask = tf.cast(query_mask, dtype=tf.float32), tf.cast(doc_mask, dtype=tf.float32)
        masks = tf.expand_dims(query_mask, -1) * tf.expand_dims(doc_mask, -2)  # (B, n_layer, Q, D)

        simmats_dot = tf.linalg.matmul(query_reps, doc_reps, transpose_b=True)  # (B, n_layer, Q, D)
        query_norm = tf.expand_dims(tf.norm(query_reps, axis=-1), -1)  # (B, n_layers, Q, 1)
        doc_norm = tf.expand_dims(tf.norm(doc_reps, axis=-1), -2)  # (B, n_layers, 1, D)
        simmats = masks * tf.math.divide(simmats_dot, (tf.multiply(query_norm, doc_norm) + 1e-6))  # (B, n_layers, Q, D)

        kernel_features = [tf.reduce_sum(masks * k.call(simmats), -1) for k in self.kernels]  # (B, n_layer, Q) * K
        kernel_features = [query_mask * tf.math.log(1e-6 + kf) for kf in kernel_features]  # (B, n_layer, Q) * K
        kernel_features = [tf.reduce_sum(kf, -1) for kf in kernel_features]  # (B, n_layer) * K
        kernel_features = tf.concat(kernel_features, axis=-1)  # (B, n_layer * K)
        return kernel_features


class TFBERTCedr_Class(tf.keras.layers.Layer):
    def __init__(self, extractor, config, *args, **kwargs):
        super(TFBERTCedr_Class, self).__init__(*args, **kwargs)
        dropout_rate = 0.1
        not_vbert = (config["modeltype"] != "vbert")

        self.extractor = extractor
        self.config = config
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.bert = TFBertModel.from_pretrained(
            config["pretrained"],
            output_hidden_states=True,
            hidden_dropout_prob=dropout_rate,
        )
        self.classifier = tf.keras.layers.Dense(
            2,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name="classifier",
        )
        if not_vbert:
            self.nir = CEDRKNRM(train_kernels=config["knrm_trainkernel"])

    def parse_bert_output(self, bert_output, masks):
        """
        (B, 12, T, H) -> (B, 12, Q, H), (B, 12, Q), (B, 12, D, H), (B, 12, D)
        where B = batch_size * num_passage
        """
        qlen = self.extractor.config["maxqlen"]
        num_passage = self.extractor.config["numpassages"]

        def unbatch_psg(x, mode):
            if len(x.shape) == 2:  # mask
                x = tf.expand_dims(x, axis=1)  # from (B, T) to (B, 1, T)

            batch_size = int(x.shape[0] // num_passage)
            x_list = [x[i*batch_size:(i+1)*batch_size] for i in range(num_passage)]
            if mode == "concat":
                return tf.concat(x_list, axis=2)  # -> (B, 1, T * n_psg) or (B, n_layer, T * n_psg, H)
            elif mode == "avg":
                return tf.reduce_mean(tf.stack(x_list, axis=1), axis=1)
            elif mode == "first":
                return x_list[0]

            raise ValueError(f"Unexpected mode: {mode}")

        queries, docs = bert_output[:, :, 1:qlen+1, :], bert_output[:, :, qlen+2:-1, :]
        query_mask, doc_mask = masks[:, 1:qlen+1], masks[:, qlen+2:-1]
        queries, query_mask = [unbatch_psg(x, mode="first") for x in [queries, query_mask]]
        docs, doc_mask = [unbatch_psg(x, mode="concat") for x in [docs, doc_mask]]
        return queries, docs, query_mask, doc_mask

    def call(self, x, **kwargs):
        """ Returns logits of shape [2] """
        doc_bert_input, doc_mask, doc_seg = x[0], x[1], x[2]

        outputs = self.bert(doc_bert_input, attention_mask=doc_mask, token_type_ids=doc_seg, output_hidden_states=True)
        pooled_output = outputs[1]  # (B * num_passage, H)
        _, n_hidden = pooled_output.shape
        pooled_output = tf.reshape(pooled_output, (-1, self.extractor.config["numpassages"], n_hidden))
        pooled_output = tf.reduce_mean(pooled_output, axis=1)  # (B, H)

        if self.config["modeltype"] == "vbert":
            cedr_output = pooled_output
        else:
            all_layer_output = outputs[2]  # size: 13
            all_layer_output = tf.stack(all_layer_output, axis=1)  # (B, n_layers, H)
            parsed_inputs = self.parse_bert_output(all_layer_output, doc_mask)  # [queries, docs, query_mask, doc_mask]
            nir_output = self.nir.call(parsed_inputs)  # (B, Q, H) -> (B, K)
            cedr_output = nir_output if self.config["modeltype"] == "nir" else tf.concat([pooled_output, nir_output], axis=-1)

        cedr_output = self.dropout(cedr_output, training=kwargs.get("training", False))
        logits = self.classifier(cedr_output)  # (B, config.num_labels)
        return logits

    def predict_step(self, data, **kwargs):
        """
        Scores each passage and applies max pooling over it.
        """
        # prepare input
        posdoc_bert_input, posdoc_mask, posdoc_seg, negdoc_bert_input, negdoc_mask, negdoc_seg = data
        batch_size = tf.shape(posdoc_bert_input)[0]
        num_passages = self.extractor.config["numpassages"]
        maxseqlen = self.extractor.config["maxseqlen"]

        posdoc_bert_input = tf.reshape(posdoc_bert_input, [batch_size * num_passages, maxseqlen])
        posdoc_mask = tf.reshape(posdoc_mask, [batch_size * num_passages, maxseqlen])
        posdoc_seg = tf.reshape(posdoc_seg, [batch_size * num_passages, maxseqlen])

        # feed input to model
        passage_scores = self.call((posdoc_bert_input, posdoc_mask, posdoc_seg), training=False, **kwargs)[:, 1]
        tf.debugging.assert_equal(tf.shape(passage_scores), batch_size)
        return passage_scores

    def score(self, x, **kwargs):
        return self.predict_step(x)

    def score_pair(self, x, **kwargs):
        posdoc_bert_input, posdoc_mask, posdoc_seg, negdoc_bert_input, negdoc_mask, negdoc_seg = x

        batch_size = tf.shape(posdoc_bert_input)[0]
        num_passages = self.extractor.config["numpassages"]
        maxseqlen = self.extractor.config["maxseqlen"]

        posdoc_bert_input = tf.reshape(posdoc_bert_input, [batch_size * num_passages, maxseqlen])
        posdoc_mask = tf.reshape(posdoc_mask, [batch_size * num_passages, maxseqlen])
        posdoc_seg = tf.reshape(posdoc_seg, [batch_size * num_passages, maxseqlen])

        negdoc_bert_input = tf.reshape(negdoc_bert_input, [batch_size * num_passages, maxseqlen])
        negdoc_mask = tf.reshape(negdoc_mask, [batch_size * num_passages, maxseqlen])
        negdoc_seg = tf.reshape(negdoc_seg, [batch_size * num_passages, maxseqlen])

        pos_score = self.call((posdoc_bert_input, posdoc_mask, posdoc_seg), **kwargs)[:, 1]
        neg_score = self.call((negdoc_bert_input, negdoc_mask, negdoc_seg), **kwargs)[:, 1]

        return pos_score, neg_score


@Reranker.register
class TFBERTCedr(Reranker):
    module_name = "TFBERTCedr"

    dependencies = [
        Dependency(key="extractor", module="extractor", name="bertpassage"),
        Dependency(key="trainer", module="trainer", name="tensorflow"),
    ]

    config_spec = [
        ConfigOption("pretrained", "bert-base-uncased", "Hugging face transformer pretrained model"),
        ConfigOption("modeltype", "vbert", "which type of bert model to run. Options: vbert, nir, cedr"),
        ConfigOption("nirmodel", "KRNM", "which Neural IR model to to integrate with bert. Options: KNRM, DRMM, PACRR"),

        # for knrm:
        ConfigOption(f"knrm_trainkernel", False, "Whether to train KNRM kernel.")
    ]

    def build_model(self):
        self.model = TFBERTCedr_Class(self.extractor, self.config)
        return self.model
