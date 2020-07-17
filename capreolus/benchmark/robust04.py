import copy
import math
import json
import random
from collections import defaultdict

from capreolus import ConfigOption, Dependency, constants
from capreolus.utils.trec import load_qrels
from capreolus.utils.loginit import get_logger

from . import Benchmark

logger = get_logger(__name__)
PACKAGE_PATH = constants["PACKAGE_PATH"]


@Benchmark.register
class Robust04(Benchmark):
    """ Robust04 benchmark using the title folds from Huston and Croft. [1] Each of these is used as the test set.
        Given the remaining four folds, we split them into the same train and dev sets used in recent work. [2]

        [1] Samuel Huston and W. Bruce Croft. 2014. Parameters learned in the comparison of retrieval models using term dependencies. Technical Report.

        [2] Sean MacAvaney, Andrew Yates, Arman Cohan, Nazli Goharian. 2019. CEDR: Contextualized Embeddings for Document Ranking. SIGIR 2019.
    """

    module_name = "robust04"
    dependencies = [Dependency(key="collection", module="collection", name="robust04")]
    qrel_file = PACKAGE_PATH / "data" / "qrels.robust2004.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.robust04.301-450.601-700.txt"
    fold_file = PACKAGE_PATH / "data" / "rob04_cedr_folds.json"
    query_type = "title"


@Benchmark.register
class Robust04Yang19(Benchmark):
    """Robust04 benchmark using the folds from Yang et al. [1]

    [1] Wei Yang, Kuang Lu, Peilin Yang, and Jimmy Lin. 2019. Critically Examining the "Neural Hype": Weak Baselines and the Additivity of Effectiveness Gains from Neural Ranking Models. SIGIR 2019.
    """

    module_name = "robust04.yang19"
    dependencies = [Dependency(key="collection", module="collection", name="robust04")]
    qrel_file = PACKAGE_PATH / "data" / "qrels.robust2004.txt"
    topic_file = PACKAGE_PATH / "data" / "topics.robust04.301-450.601-700.txt"
    fold_file = PACKAGE_PATH / "data" / "rob04_yang19_folds.json"
    query_type = "title"


@Benchmark.register
class SampledRobust04(Robust04Yang19):
    module_name = "sampled_robust04"
    file_fn = PACKAGE_PATH / "data" / module_name

    config_spec = [
        ConfigOption("mode", "shallow", "which sampling mode to use: shallow or deep"),
        ConfigOption("rate", 1.0, "sampling rate: fraction number between 0 to 1"),
    ]

    @staticmethod
    def get_judgement_stat(qrels, type="sum"):
        if type == "sum":
            return sum([len(docs) for q, docs in qrels.items()])
        elif type == "avg":
            return sum([len(docs) for q, docs in qrels.items()]) / len(qrels)

    @staticmethod
    def get_pos_neg_ratio(qrels):
        n_pos = sum([len([d for d, label in docs.items() if label > 0]) for q, docs in qrels.items()])
        n_neg = sum([len([d for d, label in docs.items() if label == 0]) for q, docs in qrels.items()])
        return n_pos / n_neg

    @staticmethod
    def contain_empty_label(docs, docs2label):
        pos_doc = [docid for docid in docs if docs2label[docid] > 0]
        neg_doc = [docid for docid in docs if docs2label[docid] == 0]
        if pos_doc and neg_doc:
            return False
        return True

    @property
    def unsampled_qrels(self):
        if not hasattr(self, "_unsampled_qrels"):
            self._unsampled_qrels = load_qrels(self.unsampled_qrel_file)
        return self._unsampled_qrels

    @property
    def unsampled_folds(self):
        if not hasattr(self, "_unsampled_folds"):
            self._unsampled_folds = json.load(open(self.unsampled_fold_file, "rt"))
        return self._unsampled_folds

    def build(self):
        mode, rate = self.config["mode"], self.config["rate"]
        self.unsampled_qrel_file = self.qrel_file
        self.unsampled_fold_file = self.fold_file
        self.qrel_file = self.file_fn / (f"{mode}.%.2f.qrels.txt"%rate)
        self.fold_file = self.file_fn / (f"{mode}.%.2f.fold.json"%rate)
        self.download_if_missing()

    def prune_redundant_judgement(self, sampled_qrels, n_expected_judgement, mode):
        """ keep same number of judgements in qrels while try to keep the pos & neg label ratio same as before """
        # expected_pos_neg_rate = self.get_pos_neg_ratio(self.qrels)  # TODO: add pos/neg ratio

        if mode not in ["deep", "shallow"]:
            raise ValueError(f"Unexpected mode: {mode}, should be one of 'deep', 'shallow']")

        n_judgement = self.get_judgement_stat(sampled_qrels, "sum")
        n_remove = n_judgement - n_expected_judgement
        assert n_remove >= 0, f"expect {n_expected_judgement} judgements but the input contains only {n_judgement} judegment. current mode: {mode}, rate: {self.config['rate']}"
        logger.info(f"{n_remove} more judgements to remove")
        # if n_judgement < n_expected_judgement:
        #     logger.warning(f"The qrels are already oversampled")

        pruned_qrels = copy.deepcopy(sampled_qrels)
        qid_n_judgement = [(qid, len(docs)) for qid, docs in pruned_qrels.items()]
        qid_n_judgement = sorted(qid_n_judgement, key=lambda q_n: q_n[1])  # sort qid according to num_judgement

        def gen_qid(revert=False):
            while True:
                for qid, n in qid_n_judgement[::-1] if revert else qid_n_judgement:
                    yield qid, n

        if mode == "deep":  # higher priority to trim query
            smallest_qid, n1 = next(gen_qid(revert=False))
            random_qid, n2 = random.sample(qid_n_judgement, 1)[0]
            if n2 < n_remove:
                n_remove -= n2
                del pruned_qrels[random_qid]
            else:
                assert n1 <= n_remove
                n_remove -= n1
                del pruned_qrels[smallest_qid]

        for qid, n in gen_qid(revert=True):  # document from queries with more judgement will be dropped
            if n_remove == 0:
                break

            if not pruned_qrels[qid]:  # has already been emptied
                if mode == "shallow":
                    raise ValueError(f"Lost {qid} during pruning")
                else:
                    logger.warning(f"{qid} is removed during qrel pruning at {mode} mode. "
                                   f"This is expected if {qid} is the only query removed at this stage")
                    continue

            doc_to_drop = random.sample(pruned_qrels[qid].keys(), 1)[0]
            if self.contain_empty_label(  # don't remove if we run out of either pos or neg judged doc
                    docs=[d for d in pruned_qrels[qid] if d != doc_to_drop],
                    docs2label=pruned_qrels[qid]):
                logger.info(f"Potential empty pos/neg is detected, skip dropping qid {qid} "
                            f"(#judgement left: {len(pruned_qrels[qid])})")
                continue

            del pruned_qrels[qid][doc_to_drop]
            n_remove -= 1

        n_after_judgement = self.get_judgement_stat(pruned_qrels, "sum")
        assert n_after_judgement == n_expected_judgement, f"expect {n_expected_judgement} yet got {n_after_judgement}"
        return pruned_qrels

    def download_if_missing(self):
        """ prepare sampled files from Robust04 qrels"""
        if self.qrel_file.exists() and self.fold_file.exists():
            return

        self.qrel_file.parent.mkdir(parents=True, exist_ok=True)
        self.fold_file.parent.mkdir(parents=True, exist_ok=True)

        mode, rate, qrels = self.config["mode"], self.config["rate"], self.unsampled_qrels

        if rate <= 0 or rate > 1:
            raise ValueError(f"Sampling rate out of range: expect it from (0, 1], yet got {rate}")

        if rate == 1:
            self.qrel_file = self.unsampled_qrel_file
            self.fold_file = self.unsampled_fold_file
            logger.warning(f"No sampling is performed since sampling rate is set to {rate}")
            return

        sampled_qrels = defaultdict(dict)
        if mode == "deep":  # sample `rate` percent queries from qrels.
            for fold, split in self.unsampled_folds.items():
                # sampling within each fold's test qids s.t. the train qids in each would be kept uniformly
                test_qids = split["predict"]["test"]
                for qid in copy.deepcopy(test_qids):
                    if qid not in qrels:
                        logger.warning(f"{qid} unfound in qrels, removed")
                        test_qids.remove(qid)

                n_expected_qids = math.ceil(len(test_qids) * rate) + 1  # add one to avoid over-trim
                sampled_qids = random.sample(test_qids, n_expected_qids)
                sampled_qrels.update({qid: qrels[qid] for qid in sampled_qids})

        elif mode == "shallow":  # sample `rate` percent of documents from qrels
            for qid, docs in self.unsampled_qrels.items():
                n_expected_docids = math.ceil(len(docs) * rate)
                assert n_expected_docids > 0, f"{qid} has zero document after sampling with rate {rate}"
                sampled_docids = random.sample(docs.keys(), n_expected_docids)

                if self.contain_empty_label(sampled_docids, docs):  # skip if run out of either pos or neg judged doc
                    logger.info(f"Potential empty pos/neg is detected, skip dropping qid {qid}"
                                f"(#judgement left: {len(docs)}")
                    sampled_qrels[qid] = docs
                else:
                    sampled_qrels[qid] = {docid: docs[docid] for docid in sampled_docids}

        else:
            raise ValueError(f"Unexpected sampling mode: {mode}")

        # prune the sampled qrels s.t. shallow and deep sampling would keep same number of judgements
        n_judgement = self.get_judgement_stat(qrels, "sum")
        n_expected_judgement = math.floor(rate * n_judgement)
        sampled_qrels = self.prune_redundant_judgement(sampled_qrels, n_expected_judgement, mode)
        n_qids, n_avg_doc, n_sampled_judgement = \
            len(sampled_qrels), self.get_judgement_stat(sampled_qrels, "avg"), self.get_judgement_stat(sampled_qrels, "sum")
        logger.info(f"After {mode} sampling with rate={rate}: \n"
                    f"{n_qids} queries with average {n_avg_doc} judged documents are kept;\n"
                    f"Total judgement number: {n_sampled_judgement}/{n_judgement}; \n"
                    f"pos/neg ratio: {self.get_pos_neg_ratio(sampled_qrels)}\n")

        # update train and dev qids in each fold accordingly
        new_folds = {}
        for fold_name, split in self.unsampled_folds.items():
            train_qids, dev_qids = split["train_qids"], split["predict"]["dev"]
            new_folds[fold_name] = {
                "train_qids": [qid for qid in train_qids if qid in sampled_qrels],
                "predict": {
                    "dev": [qid for qid in dev_qids if qid in sampled_qrels],
                    "test": split["predict"]["test"]
                }
            }

        # save to disk
        with open(self.qrel_file, "w") as f:
            for qid in sorted(sampled_qrels.keys(), key=lambda x: int(x)):
                for docid, label in sampled_qrels[qid].items():
                    f.write(f"{qid} Q0 {docid} {label}\n")
        json.dump(new_folds, open(self.fold_file, "w"))
