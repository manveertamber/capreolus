import pickle
import os
import tensorflow as tf
import numpy as np
from collections import defaultdict
import random
from profane import ConfigOption
from profane.base import Dependency
from tqdm import tqdm

from capreolus.extractor import Extractor
from capreolus import get_logger
from capreolus.utils.common import padlist
from capreolus.utils.exceptions import MissingDocError

logger = get_logger(__name__)


@Extractor.register
class BertPassage(Extractor):
    module_name = "bertpassage"
    dependencies = [
        Dependency(
            key="index", module="index", name="anserini", default_config_overrides={"indexstops": True, "stemmer": "none"}
        ),
        Dependency(key="tokenizer", module="tokenizer", name="berttokenizer"),
    ]

    pad = 0
    pad_tok = "[PAD]"

    config_spec = [
        ConfigOption("maxseqlen", 256, "Maximum input length for BERT"),
        ConfigOption("usecache", False, "Should the extracted features be cached?"),
        ConfigOption("passagelen", 150, "Length of the extracted passage"),
        ConfigOption("stride", 100, "Stride"),
        ConfigOption("numpassages", 16, "Number of passages per document"),
        ConfigOption(
            "prob",
            0.1,
            "The probability that a passage from the document will be used for training (the first passage is always used)",
        ),
    ]

    def load_state(self, qids, docids):
        cache_fn = self.get_state_cache_file_path(qids, docids)
        with open(cache_fn, "rb") as f:
            state_dict = pickle.load(f)
            self.qid2toks = state_dict["qid2toks"]
            self.docid2passages = state_dict["docid2passages"]

    def cache_state(self, qids, docids):
        os.makedirs(self.get_cache_path(), exist_ok=True)
        with open(self.get_state_cache_file_path(qids, docids), "wb") as f:
            state_dict = {"qid2toks": self.qid2toks, "docid2passages": self.docid2passages}
            pickle.dump(state_dict, f, protocol=-1)

    def get_tf_feature_description(self):
        feature_description = {
            "posdoc": tf.io.FixedLenFeature([], tf.string),
            "posdoc_mask": tf.io.FixedLenFeature([], tf.string),
            "posdoc_seg": tf.io.FixedLenFeature([], tf.string),
            "negdoc": tf.io.FixedLenFeature([], tf.string),
            "negdoc_mask": tf.io.FixedLenFeature([], tf.string),
            "negdoc_seg": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.string),
        }

        return feature_description

    def create_tf_train_feature(self, sample):
        num_passages = self.config["numpassages"]

        def _bytes_feature(value):
            """Returns a bytes_list from a string / byte."""
            if isinstance(value, type(tf.constant(0))):  # if value ist tensor
                value = value.numpy()  # get value of tensor
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        posdoc, negdoc, negdoc_id = sample["posdoc"], sample["negdoc"], sample["negdocid"]
        posdoc_mask, posdoc_seg, negdoc_mask, negdoc_seg = (
            sample["posdoc_mask"],
            sample["posdoc_seg"],
            sample["negdoc_mask"],
            sample["negdoc_seg"],
        )
        label = sample["label"]
        features = []

        for i in range(num_passages):
            # Always use the first passage, then sample from the remaining passages
            if i > 0 and random.random() > self.config["prob"]:
                continue

            bert_input_line = posdoc[i]
            bert_input_line = " ".join(self.tokenizer.bert_tokenizer.convert_ids_to_tokens(list(bert_input_line)))
            passage = bert_input_line.split("[SEP]")[-2]

            # Ignore empty passages as well
            if passage.strip() == "[PAD]":
                continue

            feature = {
                "posdoc": _bytes_feature(tf.io.serialize_tensor(posdoc[i])),
                "posdoc_mask": _bytes_feature(tf.io.serialize_tensor(posdoc_mask[i])),
                "posdoc_seg": _bytes_feature(tf.io.serialize_tensor(posdoc_seg[i])),
                "negdoc": _bytes_feature(tf.io.serialize_tensor(negdoc[i])),
                "negdoc_mask": _bytes_feature(tf.io.serialize_tensor(negdoc_mask[i])),
                "negdoc_seg": _bytes_feature(tf.io.serialize_tensor(negdoc_seg[i])),
                "label": _bytes_feature(tf.io.serialize_tensor(label[i])),
            }
            features.append(feature)

        return features

    def create_tf_dev_feature(self, sample):
        """
        sample - output from self.id2vec()
        return - a tensorflow feature
        """
        posdoc, negdoc, negdoc_id = sample["posdoc"], sample["negdoc"], sample["negdocid"]
        posdoc_mask, posdoc_seg, negdoc_mask, negdoc_seg = (
            sample["posdoc_mask"],
            sample["posdoc_seg"],
            sample["negdoc_mask"],
            sample["negdoc_seg"],
        )
        label = sample["label"]

        def _bytes_feature(value):
            """Returns a bytes_list from a string / byte."""
            if isinstance(value, type(tf.constant(0))):  # if value ist tensor
                value = value.numpy()  # get value of tensor
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        feature = {
            "posdoc": _bytes_feature(tf.io.serialize_tensor(posdoc)),
            "posdoc_mask": _bytes_feature(tf.io.serialize_tensor(posdoc_mask)),
            "posdoc_seg": _bytes_feature(tf.io.serialize_tensor(posdoc_seg)),
            "negdoc": _bytes_feature(tf.io.serialize_tensor(negdoc)),
            "negdoc_mask": _bytes_feature(tf.io.serialize_tensor(negdoc_mask)),
            "negdoc_seg": _bytes_feature(tf.io.serialize_tensor(negdoc_seg)),
            "label": _bytes_feature(tf.io.serialize_tensor(label)),
        }

        return [feature]

    def parse_tf_train_example(self, example_proto):
        feature_description = self.get_tf_feature_description()
        parsed_example = tf.io.parse_example(example_proto, feature_description)

        def parse_tensor_as_int(x):
            parsed_tensor = tf.io.parse_tensor(x, tf.int64)
            parsed_tensor.set_shape([self.config["maxseqlen"]])

            return parsed_tensor

        def parse_label_tensor(x):
            parsed_tensor = tf.io.parse_tensor(x, tf.float32)
            parsed_tensor.set_shape([2])

            return parsed_tensor

        posdoc = tf.map_fn(parse_tensor_as_int, parsed_example["posdoc"], dtype=tf.int64)
        posdoc_mask = tf.map_fn(parse_tensor_as_int, parsed_example["posdoc_mask"], dtype=tf.int64)
        posdoc_seg = tf.map_fn(parse_tensor_as_int, parsed_example["posdoc_seg"], dtype=tf.int64)
        negdoc = tf.map_fn(parse_tensor_as_int, parsed_example["negdoc"], dtype=tf.int64)
        negdoc_mask = tf.map_fn(parse_tensor_as_int, parsed_example["negdoc_mask"], dtype=tf.int64)
        negdoc_seg = tf.map_fn(parse_tensor_as_int, parsed_example["negdoc_seg"], dtype=tf.int64)
        label = tf.map_fn(parse_label_tensor, parsed_example["label"], dtype=tf.float32)

        return (posdoc, posdoc_mask, posdoc_seg, negdoc, negdoc_mask, negdoc_seg), label

    def parse_tf_dev_example(self, example_proto):
        feature_description = self.get_tf_feature_description()
        parsed_example = tf.io.parse_example(example_proto, feature_description)

        def parse_tensor_as_int(x):
            parsed_tensor = tf.io.parse_tensor(x, tf.int64)
            parsed_tensor.set_shape([self.config["numpassages"], self.config["maxseqlen"]])

            return parsed_tensor

        def parse_label_tensor(x):
            parsed_tensor = tf.io.parse_tensor(x, tf.float32)
            parsed_tensor.set_shape([self.config["numpassages"], 2])

            return parsed_tensor

        posdoc = tf.map_fn(parse_tensor_as_int, parsed_example["posdoc"], dtype=tf.int64)
        posdoc_mask = tf.map_fn(parse_tensor_as_int, parsed_example["posdoc_mask"], dtype=tf.int64)
        posdoc_seg = tf.map_fn(parse_tensor_as_int, parsed_example["posdoc_seg"], dtype=tf.int64)
        negdoc = tf.map_fn(parse_tensor_as_int, parsed_example["negdoc"], dtype=tf.int64)
        negdoc_mask = tf.map_fn(parse_tensor_as_int, parsed_example["negdoc_mask"], dtype=tf.int64)
        negdoc_seg = tf.map_fn(parse_tensor_as_int, parsed_example["negdoc_seg"], dtype=tf.int64)
        label = tf.map_fn(parse_label_tensor, parsed_example["label"], dtype=tf.float32)

        return (posdoc, posdoc_mask, posdoc_seg, negdoc, negdoc_mask, negdoc_seg), label

    def _build_vocab(self, qids, docids, topics):
        if self.is_state_cached(qids, docids) and self.config["usecache"]:
            self.load_state(qids, docids)
            logger.info("Vocabulary loaded from cache")
        else:
            logger.info("Building bertpassage vocabulary")
            tokenize = self.tokenizer.tokenize
            get_doc = self.index.get_doc
            self.docid2passages = {}

            # TODO: Move this to a method
            for docid in tqdm(docids, "extract passages"):
                # Naive tokenization based on white space
                doc = get_doc(docid).split()
                passages = []
                numpassages = self.config["numpassages"]
                for i in range(0, len(doc), self.config["stride"]):
                    if i >= len(doc):
                        assert len(passages) > 0, f"no passage can be built from empty document {doc}"
                        # logger.warning(f"document failed to fill {numpassages} passages, got {len(passages)} only")
                        break
                    else:
                        # passage = padlist(doc[i: i + self.config["passagelen"]], padlen=self.config["passagelen"], pad_token=self.pad_tok)
                        passage = doc[i : i + self.config["passagelen"]]

                    # N.B: The passages are not bert tokenized.
                    passages.append(tokenize(" ".join(passage)))

                n_actual_passages = len(passages)

                # hard-coded condition from original bert maxp paper
                if n_actual_passages > numpassages:
                    passages = [passages[0]] + random.sample(passages[1:-1], numpassages - 2) + [passages[-1]]
                else:
                    for _ in range(numpassages - n_actual_passages):
                        # randomly use one of previous passages when the document is exhausted
                        passages.append(["[PAD]"])

                assert len(passages) == self.config["numpassages"]
                self.docid2passages[docid] = passages

            self.qid2toks = {qid: tokenize(topics[qid]) for qid in tqdm(qids, desc="querytoks")}

            self.cache_state(qids, docids)

    def exist(self):
        return hasattr(self, "docid2passages") and len(self.docid2passages)

    def preprocess(self, qids, docids, topics):
        if self.exist():
            return

        self.index.create_index()
        self.qid2toks = defaultdict(list)
        self.docid2passages = None

        self._build_vocab(qids, docids, topics)

    def id2vec(self, qid, posid, negid=None, label=None):
        assert label is not None

        tokenizer = self.tokenizer
        maxseqlen = self.config["maxseqlen"]

        query_toks = self.qid2toks[qid]
        pos_bert_inputs = []
        pos_bert_masks = []
        pos_bert_segs = []

        # N.B: The passages in self.docid2passages are not bert tokenized
        pos_passages = self.docid2passages[posid]
        for tokenized_passage in pos_passages:
            input_line = ["[CLS]"] + query_toks + ["[SEP]"] + tokenized_passage + ["[SEP]"]
            if len(input_line) > maxseqlen:
                input_line = input_line[:maxseqlen]
                input_line[-1] = "[SEP]"

            padded_input_line = padlist(input_line, padlen=self.config["maxseqlen"], pad_token=self.pad_tok)
            pos_bert_masks.append([1] * len(input_line) + [0] * (len(padded_input_line) - len(input_line)))
            pos_bert_segs.append([0] * (len(query_toks) + 2) + [1] * (len(padded_input_line) - len(query_toks) - 2))
            pos_bert_inputs.append(tokenizer.convert_tokens_to_ids(padded_input_line))

        # TODO: Rename the posdoc key in the below dict to 'pos_bert_input'
        data = {
            "posdocid": posid,
            "posdoc": np.array(pos_bert_inputs, dtype=np.long),
            "posdoc_mask": np.array(pos_bert_masks, dtype=np.long),
            "posdoc_seg": np.array(pos_bert_segs, dtype=np.long),
            "negdocid": "",
            "negdoc": np.zeros((self.config["numpassages"], self.config["maxseqlen"]), dtype=np.long),
            "negdoc_mask": np.zeros((self.config["numpassages"], self.config["maxseqlen"]), dtype=np.long),
            "negdoc_seg": np.zeros((self.config["numpassages"], self.config["maxseqlen"]), dtype=np.long),
            "label": np.repeat(np.array([label], dtype=np.float32), self.config["numpassages"], 0),
        }

        if negid:
            neg_bert_inputs = []
            neg_bert_masks = []
            neg_bert_segs = []
            neg_passages = self.docid2passages[negid]
            for tokenized_passage in neg_passages:
                input_line = ["[CLS]"] + query_toks + ["[SEP]"] + tokenized_passage + ["[SEP]"]
                if len(input_line) > maxseqlen:
                    input_line = input_line[:maxseqlen]
                    input_line[-1] = "[SEP]"

                padded_input_line = padlist(input_line, padlen=self.config["maxseqlen"], pad_token=self.pad_tok)
                neg_bert_masks.append([1] * len(input_line) + [0] * (len(padded_input_line) - len(input_line)))
                neg_bert_segs.append([0] * (len(query_toks) + 2) + [1] * (len(padded_input_line) - len(query_toks) - 2))
                neg_bert_inputs.append(tokenizer.convert_tokens_to_ids(padded_input_line))

            if not neg_bert_inputs:
                raise MissingDocError(qid, negid)

            data["negdocid"] = negid
            data["negdoc"] = np.array(neg_bert_inputs, dtype=np.long)
            data["negdoc_mask"] = np.array(neg_bert_masks, dtype=np.long)
            data["negdoc_seg"] = np.array(neg_bert_segs, dtype=np.long)

        return data
