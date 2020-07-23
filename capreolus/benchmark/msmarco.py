import math
import gzip
import json
import random

from capreolus import constants, ConfigOption, Dependency, constants
from capreolus.utils.common import download_file, remove_newline
from capreolus.utils.trec import topic_to_trectxt

PACKAGE_PATH = constants["PACKAGE_PATH"]
from . import Benchmark


# @Benchmark.register
# class MSMarcoPassage(Benchmark):
#     module_name = "msmarcopassage"
#     dependencies = [Dependency(key="collection", module="collection", name="msmarco")]
#
#     qrel_file = PACKAGE_PATH / "data" / "qrels.msmarcopassage.txt"
#     topic_file = PACKAGE_PATH / "data" / "topics.msmarcopassage.txt"
#     fold_file = PACKAGE_PATH / "data" / "msmarcopassage.folds.json"
#     query_type = "title"
#

@Benchmark.register
class MSMarcoDoc(Benchmark):
    """ Currently only test fold query from TREC-DL 2019 is used """
    module_name = "msmarcodoc"
    dependencies = [Dependency(key="collection", module="collection", name="msmarco")]

    qrel_file = PACKAGE_PATH / "data" / "qrels.msmarcodoc.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.msmarcodoc.txt"
    fold_file = PACKAGE_PATH / "data" / "msmarcodoc.folds.json"
    query_type = "title"

    @staticmethod
    def prepare_fold(qids):
        """ Randomly split qids into 5 folds for cross validation """
        fold_size = math.ceil(len(qids) / 5)
        random.shuffle(qids)
        folds = []
        for i in range(5):
            folds.append(qids[i*fold_size:(i+1)*fold_size])
        return folds

    def build(self):
        self.qrel_file.parent.mkdir(exist_ok=True)
        self.download_if_missing()

    def download_if_missing(self):
        if all([f.exists() for f in [self.qrel_file, self.topic_file, self.fold_file]]):
            return

        qrel_url = "https://trec.nist.gov/data/deep/2019qrels-docs.txt"
        if not self.qrel_file.exists():
            download_file(qrel_url, self.qrel_file)

        topic_url = "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz"
        tmp_topic_fn = self.get_cache_path() / "tmp" / "msmarco-test2019-queries.tsv.gz"
        if not tmp_topic_fn.exists():
            tmp_topic_fn.parent.mkdir(exist_ok=True, parents=True)
            download_file(topic_url, tmp_topic_fn)
        qids = set()
        with gzip.open(tmp_topic_fn, "r") as fin, open(self.topic_file, "w") as fout:
            for line in fin:
                line = line.decode().strip()
                qid, query = line.split("\t")
                fout.write(topic_to_trectxt(qid, query))
                qids.add(qid)

        fold_list = self.prepare_fold(list(qids))
        folds = {}
        for i in range(5):
            fold_name = f"s{i+1}"
            train_qids = fold_list[i % 5] + fold_list[(i+1) % 5] + fold_list[(i+2) % 5]
            dev_qids, test_qids = fold_list[(i+3) % 5], fold_list[(i+4) % 5]
            folds[fold_name] = {"train_qids": train_qids, "predict": {"dev": dev_qids, "test": test_qids}}
        json.dump(folds, open(self.fold_file, "w"))

        assert all([f.exists() for f in [self.qrel_file, self.topic_file, self.fold_file]])
