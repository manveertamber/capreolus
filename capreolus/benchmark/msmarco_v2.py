import os
import math
import gzip
import json
import random

from capreolus import constants, ConfigOption, Dependency, constants
from capreolus.utils.common import download_file, remove_newline
from capreolus.utils.trec import topic_to_trectxt, load_trec_topics
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)

PACKAGE_PATH = constants["PACKAGE_PATH"]
from . import Benchmark


@Benchmark.register
class MSMarcoDoc_V2(Benchmark):
    module_name = "msdoc_v2"
    dependencies = [Dependency(key="collection", module="collection", name="msdoc_v2")]

    query_type = "title"
    config_spec = []
    use_train_as_dev = False

    data_dir = PACKAGE_PATH / "data" / "msdoc_v2"
    qrel_file = data_dir / "qrels.txt"
    topic_file = data_dir / "topics.txt"
    fold_file = data_dir / "folds.json"

    @property
    def topics(self):
        if not hasattr(self, "_topics"):
            qid_topic = [line.strip().split("\t") for line in open(self.topic_file)]
            self._topics = {
                self.query_type: {qid: topic for qid, topic in qid_topic},
            }
        return self._topics

    def build(self):
        self.download_if_missing()

    def download_if_missing(self):
        self.data_dir.mkdir(exist_ok=True, parents=True)
        if all([f.exists() for f in [self.qrel_file, self.topic_file, self.fold_file]]):
            return

        assert all([f.exists() for f in [self.qrel_file, self.topic_file]])
        def load_qid_from_topic_tsv(topic_fn):
            return [line.strip().split("\t")[0] for line in open(topic_fn)]

        print("preparing fold.json")
        train_qids = load_qid_from_topic_tsv(self.data_dir / "docv2_train_queries.tsv")
        dev_qids = load_qid_from_topic_tsv(self.data_dir / "docv2_dev_queries.tsv")
        assert len(set(train_qids) & set(dev_qids)) == 0
        folds = {
            "s1": {
                "train_qids": train_qids, 
                "predict": {
                    "dev": dev_qids, 
                    "test": dev_qids, 
        }}}
        json.dump(folds, open(self.fold_file, "w"))
