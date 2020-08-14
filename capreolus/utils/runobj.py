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
        scores = {}
        batch_runs = []
        # with Pool(10) as p:
        # scores = {
        #     q: s
        #     for qid, doc2score in tqdm(self.items(), total=273633501)
        #     for q, s in evaluator.evaluate({qid: doc2score}).items()
        # }
        total = 273633501 if not qids else len(qids)
        scores = {
            # qid: evaluator.evaluate({qid: doc2score})[qid]
            # qid: evaluator.evaluate({qid: doc2score}).get(qid, {})
            qid: eval_fn({qid: doc2score}).get(qid, {})
            for qid, doc2score in tqdm(self.items(), total=total)
            if not qid or qid in qids
        }
        # for qid, doc2score in tqdm(self.items(), total=273633501):
        #     scores[qid] = evaluator.evaluate({qid: doc2score})[qid]

        # scores = {q: s for score in scores for q, s in score}
        if False:
            pass
            # for single_run in self.items():
            #     batch_runs.append(single_run)
            #     if len(batch_runs) == 100:
            #         print("evaluating", end="\t")
            #         batch_scores = p.map(evaluator.evaluate, batch_runs)
            #         print("got result", end="\t")
            #         scores.update({q: s for single_score in batch_scores for q, s in single_score})
            #         print("combined result")
        return scores


def run_test():
    runfile = "/home/xinyu1zhang/.capreolus/results/collection-msmarcopsg/benchmark-msmarcopsg_qrelsize-small/collection-msmarcopsg/benchmark-msmarcopsg_qrelsize-small/searcher-msmarcopsg_searcher/task-rank_filter-False/searcher"
    runs = Runs(runfile)
    # for i, k in enumerate(runs):
    #     print(k)
    #     if i == 10:
    #         break

    # for i, (qid, doc2score) in enumerate(runs.items()):
    #     print(doc2score)
    #     print(qid)
    #     break

    qrelfile = "/home/xinyu1zhang/2021aaai/capreolus/capreolus/data/msmarcopsg/small/qrels.msmarcodoc.txt"
    from nirtools.ir import load_qrels
    from time import time
    import json
    import os
    from capreolus.evaluator import get_eval_runs

    qrels = load_qrels(qrelfile)
    fold_file = "/home/xinyu1zhang/2021aaai/capreolus/capreolus/data/msmarcopsg/small/msmarcodoc.folds.json"
    folds = json.load(open(fold_file))["s1"]
    dev, test = folds["predict"]["dev"], folds["predict"]["test"]

    t1 = time()
    for i, qids in enumerate([dev]):
        json_fn = f"msmarcopsgresult.{i}.json"
        if os.path.exists(json_fn):
            final_result = json.load(open(json_fn))
        else:
            eval_fn = get_eval_runs(qrels, ["mrr", "map"], dev_qids=dev, relevance_level=1)
            final_result = runs.evaluate(eval_fn, qids)
            print(len(final_result), "time: ", time() - t1)
            json.dump(final_result, open(json_fn, "w"))

        scores = [[metrics_dict.get(m, -1) for m in ["mrr", "map"]] for metrics_dict in final_result.values()]
        import numpy as np
        scores = np.array(scores).mean(axis=0).tolist()
        scores = dict(zip(["mrr", "map"], scores))
        print(scores)

run_test()
