import numpy as np
import torch

import capnp
from tqdm import tqdm
from pymagnitude import Magnitude, MagnitudeUtils

from capreolus.extractor import Extractor
from capreolus.utils.common import padlist
from capreolus.utils.loginit import get_logger
from capreolus.tokenizer import Tokenizer

logger = get_logger(__name__)  # pylint: disable=invalid-name


class BertText(Extractor):
    """ Tokenize text using bert-base-uncased's word pieces. """

    @staticmethod
    def config():
        return locals().copy()  # ignored by sacred

    def build_from_benchmark(self):
        if not all([self.collection, self.benchmark, self.index]):
            raise ValueError("The Feature class was not initialized with a collection, benchmark and index.")

        tokenizer_name, tokmodel = "bertpair", "bert-base-uncased"
        self.tokenizer = Tokenizer.ALL[tokenizer_name].get_tokenizer_instance(self.index, tokmodel=tokmodel)
        self.tokenizer.create()

        self.CLSS = [self.tokenizer.vocab["[CLS]"]]
        self.SEPS = [self.tokenizer.vocab["[SEP]"]]
        self.ONES = [1]
        self.NILS = [0]

        # BERT maximum input size is 512 - 2 SEP - CLS - query
        self.maxdoclen = min(self.pipeline_config["maxdoclen"], 512 - 3 - self.pipeline_config["maxqlen"])
        if self.maxdoclen < self.pipeline_config["maxdoclen"]:
            logger.warning("reducing maxdoclen to %s due to BERT's max input size of 512", self.maxdoclen)

        # tokenize docs and handle vocab, including embeddings for missing terms
        self.index.open()

        # the tokenizer cache expects strings, so we convert all cached docs
        self.tokenizer.cache = {k: [str(x) for x in v] for k, v in self.tokenizer.cache.items()}
        self.tokenizer.write_cache()

    def transform_qid_posdocid_negdocid(self, q_id, posdoc_id, negdoc_id=None):
        if not all([self.collection, self.benchmark, self.index]):
            raise ValueError("The Feature class was not initialized with a collection, benchmark and index.")

        queries = self.collection.topics[self.benchmark.query_type]
        posdoc_toks = self.tokenizer.tokenizedoc(posdoc_id)[: self.pipeline_config["maxdoclen"]] if posdoc_id else []
        posdoc_toks = [int(x) for x in posdoc_toks]
        negdoc_toks = self.tokenizer.tokenizedoc(negdoc_id)[: self.pipeline_config["maxdoclen"]] if negdoc_id else []
        negdoc_toks = [int(x) for x in negdoc_toks]
        query_toks = self.tokenizer.tokenize(queries[q_id])

        if len(query_toks) > self.pipeline_config["maxqlen"]:
            logger.warning(
                "query '%s' of size %s is longer than maxqlen=%s", queries[q_id], len(query_toks), self.pipeline_config["maxqlen"]
            )
            query_toks = query_toks[: self.pipeline_config["maxqlen"]]

        if not posdoc_toks or (negdoc_id and not negdoc_toks):
            logger.debug("missing docid %s and/or %s", posdoc_id, negdoc_id)
            return None

        # TODO: the padding approach seems different with BERT:
        # say there is a two-pair batch: [(q1, q2, d1, d2), (q1, d1)]
        # how BERT pad and mask the batch is: [(q1, q2, [seg], d1, d2, [seg]), (q1, [seg], d1, [seg], pad, pad)]
        qtoks, qmask = pad_and_mask(query_toks, self.pipeline_config["maxqlen"])
        qsegs = [0 for _ in qmask]
        doc_features = self.get_doc_features(posdoc_toks, "pos", qtoks, qmask, qsegs)

        if negdoc_id is not None:
            negdoc_features = self.get_doc_features(negdoc_toks, "neg", qtoks, qmask, qsegs)
            doc_features.update(negdoc_features)

        doc_features["posdocid"] = posdoc_id
        doc_features["negdocid"] = negdoc_id
        doc_features["qid"] = q_id

        return doc_features

    def get_doc_features(self, doc_toks, prefix, qtoks, qmask, qsegs):
        if not doc_toks:
            return {}

        d = {}
        dtoks, dmask = pad_and_mask(doc_toks, self.maxdoclen)
        dsegs = [1 for _ in dmask]

        toks = self.CLSS + qtoks + self.SEPS + dtoks + self.SEPS
        mask = self.ONES + qmask + self.ONES + dmask + self.ONES    # so in this way there would be an isolated 1 at the end of mask
        segs = self.NILS + qsegs + self.NILS + dsegs + self.ONES

        assert len(toks) <= 512
        assert len(toks) == len(mask)
        assert len(mask) == len(segs)

        # pads = [0 for _ in range(512 - len(toks))]


        d[prefix + "toks"] = np.array(toks)
        d[prefix + "mask"] = np.array(mask)
        d[prefix + "segs"] = np.array(segs)
        d[prefix + "qmask"] = np.array(qmask)
        d[prefix + "dmask"] = np.array(dmask)

        return d


def pad_and_mask(s, tolen):
    s = s[:tolen]

    padding = [0 for _ in range(tolen - len(s))]
    mask = [1 for _ in s] + [0 for _ in padding]
    # padding = []
    # mask = [1 for _ in s]
    return s + padding, mask
