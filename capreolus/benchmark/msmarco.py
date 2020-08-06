import os
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


@Benchmark.register
class MSMarcoPassage(Benchmark):
    module_name = "msmarcopsg"
    dependencies = [Dependency(key="collection", module="collection", name="msmarcopsg")]

    data_dir = PACKAGE_PATH / "data" / "msmarcopsg"
    query_type = "title"

    config_spec = [ConfigOption("qrelsize", "small", "Which training triplet to use: `Train Triples Small` or `Train Triples Large`")]

    @staticmethod
    def prepare_fold(qids):
        pass

    def build(self):
        data_dir = self.data_dir / self.config["qrelsize"]
        self.qrel_file = data_dir / "qrels.msmarcodoc.txt"
        self.topic_file = data_dir / "topics.msmarcodoc.txt"
        self.fold_file = data_dir / "msmarcodoc.folds.json"

        data_dir.mkdir(exist_ok=True, parents=True)

        self.download_if_missing()

    def download_if_missing(self):
        if all([f.exists() for f in [self.qrel_file, self.topic_file, self.fold_file]]):
            return

        def match_size(fn):
            if ".train." in fn:
                return True

            if self.config["qrelsize"] == "small":
                return ".small." in fn
            return ".small." not in fn

        gz_dir = self.collection.download_raw()
        queries_fn = [fn for fn in os.listdir(gz_dir) if "queries." in fn and match_size(fn)]
        qrels_fn = [fn for fn in os.listdir(gz_dir) if "qrels." in fn and match_size(fn)]  # note that qrel.test is not given

        # topic and qrel
        topic_f, qrel_f = open(self.topic_file, "w"), open(self.qrel_file, "w")
        folds = {"train": set(), "dev": set(), "eval": set()}

        for set_name in folds:
            cur_queriesfn = [fn for fn in queries_fn if f".{set_name}." in fn]
            cur_qrelfn = [fn for fn in qrels_fn if f".{set_name}." in fn]
            with open(gz_dir / cur_queriesfn[0], "r") as f:
                for line in f:
                    qid, query = line.strip().split("\t")
                    topic_f.write(topic_to_trectxt(qid, query))
                    folds[set_name].add(qid)

            if not cur_qrelfn:
                logger.warning(f"{set_name} qrel is unfound. This is expected if it is eval set. "
                               f"This is unexpected if it is train or dev set.")
                continue

            with open(gz_dir / cur_qrelfn[0], "r") as f:
                for line in f:
                    qrel_f.write(line)

        # fold
        folds = {k: list(v) for k, v in folds.items()}
        folds = {"s1": {"train_qids": folds["train"], "predict": {"dev": folds["dev"], "test": folds["eval"]}}}
        json.dump(folds, open(self.fold_file, "w"))


@Benchmark.register
class MSMarcoDoc(Benchmark):
    """ Currently only test fold query from TREC-DL 2019 is used """
    module_name = "msmarcodoc"
    dependencies = [Dependency(key="collection", module="collection", name="msmarcodoc")]
    config_spec = [
        ConfigOption("judgementtype", "deep", "Whether to use deep or shallow qrels")
    ]

    data_dir = PACKAGE_PATH / "data" / "msmarcodoc"
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
