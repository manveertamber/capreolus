import math
import gzip
import json
import random

from capreolus import constants, ConfigOption, Dependency, constants
from capreolus.utils.common import download_file, remove_newline
from capreolus.utils.trec import topic_to_trectxt, load_qrels
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)

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
    config_spec = [
        ConfigOption("judgementtype", "deep", "Whether to use deep or shallow qrels")
    ]

    data_dir = PACKAGE_PATH / "data" / "msmarco"
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
        data_dir = self.data_dir / self.config["judgementtype"]
        self.qrel_file = data_dir / "qrels.msmarcodoc.txt"
        self.topic_file = data_dir / "topics.msmarcodoc.txt"
        self.fold_file = data_dir / "msmarcodoc.folds.json"

        data_dir.mkdir(exist_ok=True, parents=True)
        self.download_if_missing()

    def download_deep_qrel_topic(self, total_judgement_number):
        qrel_url = "https://trec.nist.gov/data/deep/2019qrels-docs.txt"
        if not self.qrel_file.exists():
            download_file(qrel_url, self.qrel_file)
        qrels = load_qrels(self.qrel_file)
        qids = list(qrels.keys())
        assert sum([len(qrels[qid]) for qid in qrels]) == total_judgement_number
        return qids

    def download_shallow_qrel_topic(self, total_judgement_number):
        """ keep the first `total_judgement_number` queries from train set """
        qrel_url = "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz"
        tmp_qrels_fn = self.get_cache_path() / "tmp" / "qrels.train.tsv"
        if not tmp_qrels_fn.exists():
            tmp_qrels_fn.parent.mkdir(exist_ok=True, parents=True)
            download_file(qrel_url, tmp_qrels_fn)

        qids = set()
        with gzip.open(tmp_qrels_fn, "r") as fin, open(self.qrel_file, "w") as fout:
            for line in fin:
                if len(qids) == total_judgement_number:
                    break

                line = line.decode().strip()
                qid = line.split()[0]
                if qid in qids:  # only keep one judgement per query
                    continue
                fout.write(line)
                qids.add(qid)

        return list(qids)

    def download_topic(self, qids, topic_url):
        tmp_topic_fn = self.get_cache_path() / "tmp" / topic_url.split("/")[-1]  # "msmarco-test2019-queries.tsv.gz"
        if not tmp_topic_fn.exists():
            tmp_topic_fn.parent.mkdir(exist_ok=True, parents=True)
            download_file(topic_url, tmp_topic_fn)
        with gzip.open(tmp_topic_fn, "r") as fin, open(self.topic_file, "w") as fout:
            for line in fin:
                line = line.decode().strip()
                qid, query = line.split("\t")
                if qid in qids:
                    fout.write(topic_to_trectxt(qid, query))

    def download_if_missing(self):
        if all([f.exists() for f in [self.qrel_file, self.topic_file, self.fold_file]]):
            return

        total_judgement_number = 16258  # number of judgement contained in deep qrels, will keep the first 16258 judgement from shallow qrel for fair comparison
        if self.config["judgementtype"] == "deep":
            qids = self.download_deep_qrel_topic(total_judgement_number)
            topic_url = "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz"
        elif self.config["judgementtype"] == "shallow":
            qids = self.download_shallow_qrel_topic(total_judgement_number)
            topic_url = "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz"
        self.download_topic(qids, topic_url)

        fold_list = self.prepare_fold(list(qids))
        folds = {}
        for i in range(5):
            fold_name = f"s{i+1}"
            train_qids = fold_list[i % 5] + fold_list[(i+1) % 5] + fold_list[(i+2) % 5]
            dev_qids, test_qids = fold_list[(i+3) % 5], fold_list[(i+4) % 5]
            folds[fold_name] = {"train_qids": train_qids, "predict": {"dev": dev_qids, "test": test_qids}}
        json.dump(folds, open(self.fold_file, "w"))

        assert all([f.exists() for f in [self.qrel_file, self.topic_file, self.fold_file]])
