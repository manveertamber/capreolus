import os
from collections import defaultdict

import numpy as np
import pytrec_eval

from capreolus.searcher import Searcher
from capreolus.utils.loginit import get_logger
from capreolus.utils.runobj import Runs

logger = get_logger(__name__)

DEFAULT_METRICS = [
    "P_1",
    "P_5",
    "P_10",
    "P_20",
    "judged_10",
    "judged_20",
    "judged_200",
    "map",
    "mrr",
    "ndcg_cut_5",
    "ndcg_cut_10",
    "ndcg_cut_20",
    "recall_100",
    "recall_1000",
    "recip_rank",
]


def judged(qrels, runs, n, aggregate=True):
    qids, scores = [], []
    for q, rundocs in runs.items():
        if q not in qrels:
            logger.error(f"{q} in run files cannot be found in qrels")
            continue
        qids.append(q)

        if len(rundocs) == 0:
            scores.append(0)
            continue

        topn = sorted(rundocs.keys(), key=rundocs.get, reverse=True)[:n]
        score = sum(docid in qrels[q] for docid in topn) / len(topn)
        scores.append(score)

    if not aggregate:
        return dict(zip(qids, scores))

    return sum(scores) / len(scores) if len(scores) != 0 else 0


def _mrr(qrel, rundoc):
    """
    calculate the mrr for a list of docs from same query
    :param qrel: dict, mapping the doc id into ground truth label
    :param rundoc: dict, mapping the doc id into doc score
    :return: float, the mrr score
    """
    if (not rundoc) or (not qrel):
        return 0.

    pos_docids, pos_doc_ranks = [d for d in rundoc if qrel.get(d, 0) > 0], []
    if not pos_docids:  # or all([d not in rundoc for d in pos_docids]):
        return 0.

    rundoc = sorted(rundoc.items(), key=lambda doc_score: float(doc_score[1]), reverse=True)
    rundoc = [d for d, i in rundoc]

    pos_doc_ranks = [rundoc.index(d) + 1 for d in pos_docids]
    return 1 / min(pos_doc_ranks)


def mrr(qrels, runs, qids=None, aggregate=True):
    qids = set(qrels.keys()) & set(runs.keys()) & set(qids) if qids else set(qrels.keys()) & set(runs.keys())
    qrel_rundoc = [(qrels.get(q, {}), runs.get(q, {})) for q in qids]
    ranks = [_mrr(*qr) for qr in qrel_rundoc]

    if not aggregate:
        return dict(zip(qids, ranks))

    return sum(ranks) / len(ranks)


def aggregate_score_list(metric2score_lst):
    """
    :param metric2score_lst: a list of {metric: score, ...}
    :return: a dict {metric: score} by average each metric's score along the list
    """
    if not isinstance(metric2score_lst, list):
        metric2score_lst = list(metric2score_lst)
    metrics = sorted(list(metric2score_lst[0].keys()))
    scores = [[metric2score[m] for m in metrics] for metric2score in metric2score_lst]
    scores = np.array(scores).mean(axis=0).tolist()
    scores = dict(zip(metrics, scores))
    return scores


def get_runs_evaluator(qrels, metrics, dev_qids, relevance_level):
    assert isinstance(metrics, list)
    notrec_metrics = [m for m in metrics if m.startswith("judged_") or m in ["mrr"]]
    trec_metrics = [m for m in metrics if m not in notrec_metrics]
    dev_qrels = {qid: labels for qid, labels in qrels.items() if qid in dev_qids}

    def _eval_runs_fn(runs):
        evaluator = pytrec_eval.RelevanceEvaluator(dev_qrels, trec_metrics, relevance_level=int(relevance_level))
        # ^ move this line outside of _eval_run would cause some of the trec_metrics to be deleted after
        # the first time of evaluation
        runs = {qid: runs[qid] for qid in runs if qid in dev_qids}
        scores = evaluator.evaluate(runs)
        for metric in notrec_metrics:
            if metric.startswith('judged_'):
                n = int(metric.split("_")[1])
                cur_scores = judged(dev_qrels, runs, n, aggregate=False)
            if metric == "mrr":
                cur_scores = mrr(dev_qrels, runs, aggregate=False)
            scores = {qid: {metric: cur_scores[qid], **scores[qid]} for qid in scores}

        return scores

    return _eval_runs_fn


def search_best_run(runfile_dirs, benchmark, primary_metric, metrics=None, folds=None):
    """
    Select the runfile with respect to the specified metric

    Args:
        runfile_dirs: the directory path to all the runfiles to select from
        benchmark: Benchmark class
        primary_metric: str, metric used to select the best runfile , e.g. ndcg_cut_20, etc
        metrics: str or list, metric expected by be calculated on the best runs
        folds: str, the name of fold to select from

    Returns:
       a dict storing specified metric score and path to the corresponding runfile
    """

    if not isinstance(runfile_dirs, (list, tuple)):
        runfile_dirs = [runfile_dirs]

    metrics = [] if not metrics else ([metrics] if isinstance(metrics, str) else list(metrics))
    if primary_metric not in metrics:
        metrics = [primary_metric] + metrics

    folds = {s: benchmark.folds[s] for s in [folds]} if folds else benchmark.folds
    runfiles = [
        os.path.join(runfile_dir, f)
        for runfile_dir in runfile_dirs
        for f in os.listdir(runfile_dir)
        if (f != "done" and not os.path.isdir(os.path.join(runfile_dir, f)))
    ]

    best_scores = {fold: {primary_metric: 0, "path": None} for fold in folds}
    eval_run_fns = {fold: get_runs_evaluator(
        benchmark.qrels, [primary_metric],
        dev_qids=set(qids["predict"]["dev"]),
        relevance_level=benchmark.relevance_level
    ) for fold, qids in folds.items()}
    for runfile in runfiles:
        runs = Runs(runfile, buffer=benchmark.collection.is_large_collection)
        for fold, qids in folds.items():
            eval_run_fn = eval_run_fns[fold]
            qid2score = runs.evaluate(eval_run_fn, qids=set(qids["predict"]["dev"]))
            score = aggregate_score_list(qid2score.values())[primary_metric]
            if score > best_scores[fold][primary_metric]:
                best_scores[fold] = {primary_metric: score, "path": runfile}

    scores = {}
    for fold, score_dict in best_scores.items():
        test_qids = folds[fold]["predict"]["test"]
        runs = Runs(score_dict["path"], buffer=benchmark.collection.is_large_collection)
        test_eval_fn = get_runs_evaluator(
            benchmark.qrels, metrics,
            dev_qids=test_qids,
            relevance_level=benchmark.relevance_level
        )
        scores.update(runs.evaluate(test_eval_fn, qids=test_qids))

    scores = aggregate_score_list(scores.values())
    return {"score": scores, "path": {s: v["path"] for s, v in best_scores.items()}}


def interpolate_runs(run1, run2, qids, alpha):
    out = {}
    for qid in qids:
        out[qid] = {}

        if len(run1[qid]) == 0:
            min1, max1 = 0, 1
        else:
            min1, max1 = min(run1[qid].values()), max(run1[qid].values())

            if min1 == max1:
                min1 = 0.01 * max1 - 0.01

        if len(run2[qid]) == 0:
            min2, max2 = 0, 1
        else:
            min2, max2 = min(run2[qid].values()), max(run2[qid].values())

            if min2 == max2:
                min2 = 0.01 * max2 - 0.01

        for docid in run1[qid].keys() | run2[qid]:
            score1 = run1[qid].get(docid, min1)
            score2 = run2[qid].get(docid, min2)

            score1 = (score1 - min1) / (max1 - min1)
            score2 = (score2 - min2) / (max2 - min2)
            out[qid][docid] = alpha * score1 + (1 - alpha) * score2

    return out


def interpolated_eval(run1, run2, benchmark, primary_metric, metrics=None):
    metrics = [] if not metrics else ([metrics] if isinstance(metrics, str) else list(metrics))
    if primary_metric not in metrics:
        metrics = [primary_metric] + metrics

    test_runs = {}
    alphas = {}
    for s, v in benchmark.folds.items():
        best_metric = None
        dev_qids = set(v["predict"]["dev"])
        dev1, dev2 = run1[s]["dev"], run2[s]["dev"]

        for alpha in np.arange(0, 1.001, 0.05):
            interpolated_run = interpolate_runs(dev1, dev2, dev_qids, alpha)
            metrics = eval_runs(interpolated_run, benchmark.qrels, metrics, benchmark.relevance_level)

            if best_metric is None or metrics[primary_metric] > best_metric:
                best_metric = metrics[primary_metric]
                alphas[s] = alpha

        test_qids = set(v["predict"]["test"])
        test1, test2 = run1[s]["test"], run2[s]["test"]
        interpolated_test_run = interpolate_runs(test1, test2, test_qids, alphas[s])
        for qid in test_qids:
            assert qid not in test_runs
            test_runs[qid] = interpolated_test_run[qid].copy()

    scores = eval_runs(test_runs, benchmark.qrels, metrics, benchmark.relevance_level)
    return {"score": scores, "alphas": alphas}
