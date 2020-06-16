import tensorflow as tf
from profane import ConfigOption, Dependency

from capreolus.reranker import Reranker
from capreolus.reranker.common import RbfKernelBankTF, similarity_matrix_tf
from capreolus.reranker.base import KerasModel


class TFKNRM_Class(tf.keras.Model):
    def __init__(self, extractor, config, **kwargs):
        super(TFKNRM_Class, self).__init__(**kwargs)
        self.config = config
        self.extractor = extractor
        mus = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        sigmas = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.001]
        self.embedding = tf.keras.layers.Embedding(
            len(self.extractor.stoi), self.extractor.embeddings.shape[1], weights=[self.extractor.embeddings], trainable=False
        )
        self.kernels = RbfKernelBankTF(mus, sigmas, dim=1, requires_grad=config["gradkernels"])
        self.combine = tf.keras.layers.Dense(1, input_shape=(self.kernels.count(),))

    def get_score(self, doc_tok, query_tok, query_idf):
        query = self.embedding(query_tok)
        doc = self.embedding(doc_tok)
        batch_size, qlen, doclen = tf.shape(query)[0], tf.shape(query)[1], tf.shape(doc)[1]

        simmat = similarity_matrix_tf(query, doc, query_tok, doc_tok, self.extractor.pad)

        k = self.kernels(simmat)
        doc_k = tf.reduce_sum(k, axis=3)  # sum over document
        reshaped_simmat = tf.broadcast_to(
            tf.reshape(simmat, (batch_size, 1, qlen, doclen)), (batch_size, self.kernels.count(), qlen, doclen)
        )
        mask = tf.reduce_sum(reshaped_simmat, axis=3) != 0.0
        log_k = tf.where(mask, tf.math.log(doc_k + 1e-6), tf.cast(mask, tf.float32))
        query_k = tf.reduce_sum(log_k, axis=2)
        scores = self.combine(query_k)

        return tf.reshape(scores, [batch_size])

    def call(self, x, **kwargs):
        doc, query, query_idf = x[0], x[1], x[2]
        score = self.get_score(doc, query, query_idf)

        return score


@Reranker.register
class TFKNRM(Reranker):
    module_name = "TFKNRM"
    dependencies = [
        Dependency(key="extractor", module="extractor", name="embedtext"),
        Dependency(key="trainer", module="trainer", name="tensorflow"),
    ]
    config_spec = [
        ConfigOption("gradkernels", True, "backprop through mus and sigmas"),
        ConfigOption("finetune", False, "fine tune the embedding layer"),  # TODO check save when True
    ]

    def build_model(self):
        self.model = KerasModel(TFKNRM_Class(self.extractor, self.config), self.config)

        return self.model
