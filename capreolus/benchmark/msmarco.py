import os
import math
import gzip
import json
import random

from capreolus import constants, ConfigOption, Dependency, constants
from capreolus.utils.common import download_file, remove_newline
from capreolus.utils.trec import topic_to_trectxt, load_trec_topics
from capreolus.utils.loginit import get_logger

from . import Benchmark, IRDBenchmark

logger = get_logger(__name__)
PACKAGE_PATH = constants["PACKAGE_PATH"]


@Benchmark.register
class MSMarcoPassage(Benchmark):
    module_name = "msmarcopsg"
    dependencies = [Dependency(key="collection", module="collection", name="msmarcopsg")]

    query_type = "title"
    config_spec = []
    use_train_as_dev = False

    data_dir = PACKAGE_PATH / "data" / "msmarcopsg"
    qrel_file = data_dir / "qrels.txt"
    topic_file = data_dir / "topics.txt"
    fold_file = data_dir / "folds.json"

    def build(self):
        self.download_if_missing()

    def download_if_missing(self):
        self.data_dir.mkdir(exist_ok=True, parents=True)
        if all([f.exists() for f in [self.qrel_file, self.topic_file, self.fold_file]]):
            return

        def match_size(fn):
            if ".train." in fn:
                return True

            # if self.config["qrelsize"] == "small":
            if True:
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
                logger.warning(
                    f"{set_name} qrel is unfound. This is expected if it is eval set. "
                    f"This is unexpected if it is train or dev set."
                )
                continue

            with open(gz_dir / cur_qrelfn[0], "r") as f:
                for line in f:
                    qrel_f.write(line)

        # fold
        folds = {k: list(v) for k, v in folds.items()}
        folds = {"s1": {"train_qids": folds["train"], "predict": {"dev": folds["dev"], "test": folds["eval"]}}}
        json.dump(folds, open(self.fold_file, "w"))


@Benchmark.register
class MSMARCO_Passage_V2(IRDBenchmark):
    module_name = "mspsg_v2"
    query_type = "text"
    ird_dataset_names = ["msmarco-passage-v2/train", "msmarco-passage-v2/dev1", "msmarco-passage-v2/trec-dl-2021"]
    dependencies = [Dependency(key="collection", module="collection", name="mspsg_v2")]
    fold_file = PACKAGE_PATH / "data" / "msmarcov2_passage_title_folds.json"


@Benchmark.register
class MSMARCO_Document_V2(IRDBenchmark):
    module_name = "msdoc_v2"
    query_type = "text"
    ird_dataset_names = ["msmarco-document-v2/train", "msmarco-document-v2/dev1", "msmarco-document-v2/trec-dl-2021"]
    dependencies = [Dependency(key="collection", module="collection")]
    fold_file = PACKAGE_PATH / "data" / "msmarcov2_doc_title_folds.json"
