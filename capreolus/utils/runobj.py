import tarfile
from pathlib import Path
from multiprocessing import Pool

from capreolus.utils.common import OrderedDefaultDict
from capreolus.utils.loginit import get_logger

from tqdm import tqdm

logger = get_logger(__name__)


class Runs:
    supported_format = ["trec", "tar"]

    def __init__(self, runfile, format="trec"):
        """ TODO: support init from runs """
        if format not in self.supported_format:
            raise ValueError(f"Unrecognized format: {format}, expected to be one of {' '.join(self.supported_format)}")

        self.runfile = runfile
        self.format = format
        self.qids = set()

    @staticmethod
    def load_trec_run(fn):
        # Docids in the run file appear according to decreasing score, hence it makes sense to preserve this order
        run = OrderedDefaultDict()

        with open(fn, "rt") as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    qid, _, docid, rank, score, desc = line.split(" ")
                    run[qid][docid] = float(score)
        return run

    @staticmethod
    def write_trec_run(preds, outfn):
        count = 0
        with open(outfn, "wt") as outf:
            qids = sorted(preds.keys(), key=lambda k: int(k))
            for qid in qids:
                rank = 1
                for docid, score in sorted(preds[qid].items(), key=lambda x: x[1], reverse=True):
                    print(f"{qid} Q0 {docid} {rank} {score} capreolus", file=outf)
                    rank += 1
                    count += 1

    @staticmethod
    def from_runfile(format="trec", buffer=True):
        """
        :param format: str, input runfile format, must be one of `self.supported_format`
        :return: a Runs object
        """
        if format == "trec" and not buffer:
            pass

    def _open_runfile(self):
        return open(self.runfile) if self.format == "trec" else tarfile.open(self.runfile)

    def keys(self):
        if not self.qids:  # the first time
            f = self._open_runfile()
            for line in f:
                qid = line.split()[0]
                if qid in self.qids:
                    continue

                self.qids.add(qid)
                yield qid
        else:
            for qid in sorted(list(self.qids), key=lambda x: int(x)):
                yield qid

    def values(self):
        # assume
        pass

    def items(self):
        # assume records with same qid is same
        f = self._open_runfile()
        last_qid = -1
        doc2score = {}
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()

            if last_qid == -1:
                last_qid = qid

            if qid != last_qid:
                yield last_qid, doc2score

                last_qid = qid
                doc2score = {}

            doc2score[docid] = float(score)

        f.close()
        yield qid, doc2score  # the last item

    def __iter__(self):
        return iter(self.keys())

    def evaluate(self, eval_fn, qids=None):
        """
        :param evaluator: pytrec Evaluator
        :return: evaluated runs
        """
        # total = 273633501 if not qids else len(qids)
        scores = {
            qid: eval_fn({qid: doc2score})[qid]
            for qid, doc2score in self.items()
            if not qids or qid in qids
        }
        return scores


def run_test():
    # runfile = "/home/xinyu1zhang/.capreolus/results/collection-msmarcopsg/benchmark-msmarcopsg_qrelsize-small/collection-msmarcopsg/benchmark-msmarcopsg_qrelsize-small/searcher-msmarcopsg_searcher/task-rank_filter-False/searcher"
    runfile = "/home/xinyu1zhang/.capreolus/results/collection-robust04/benchmark-robust04/collection-robust04/index-anserini_indexstops-False_stemmer-porter/searcher-BM25_b-0.4_fields-title_hits-1000_k1-0.9/task-rank_filter-False/searcher"
    runs = Runs(runfile)
    # qrelfile = "/home/xinyu1zhang/2021aaai/capreolus/capreolus/data/msmarcopsg/small/qrels.msmarcodoc.txt"
    qrelfile = "/home/xinyu1zhang/2021aaai/capreolus/capreolus/data/qrels.robust2004.txt"

    from nirtools.ir import load_qrels
    import json
    from capreolus.evaluator import get_eval_runs
    from capreolus.evaluator import DEFAULT_METRICS
    import numpy as np

    metrics = DEFAULT_METRICS
    qrels = load_qrels(qrelfile)
    # fold_file = "/home/xinyu1zhang/2021aaai/capreolus/capreolus/data/msmarcopsg/small/msmarcodoc.folds.json"
    fold_file = "/home/xinyu1zhang/2021aaai/capreolus/capreolus/data/rob04_yang19_folds.json"
    folds = json.load(open(fold_file))

    scores_allfolds = []
    for f in folds:
        fold = folds[f]
        dev, test = fold["predict"]["dev"], fold["predict"]["test"]
        dev, test = list(set(dev) & set(qrels)), list(set(test) & set(qrels))

        for i, qids in enumerate([test]):
            json_fn = f"rob04.{i}.json"
            eval_fn = get_eval_runs(qrels, metrics, dev_qids=qids, relevance_level=1)
            final_result = runs.evaluate(eval_fn, qids)
            json.dump(final_result, open(json_fn, "w"))

            scores = [[metrics_dict.get(m, -1) for m in metrics] for metrics_dict in final_result.values()]
            scores_allfolds.extend(scores)

    from pprint import pprint
    scores_allfolds = np.array(scores_allfolds).mean(axis=0).tolist()
    pprint(dict(zip(metrics, scores_allfolds)))

run_test()
