import os
import glob
from tqdm import tqdm
from multiprocessing import Pool, get_context

import pytrec_eval
import numpy as np

from capreolus.utils.loginit import get_logger
from capreolus.searcher import Searcher

logger = get_logger(__name__)

VALID_METRICS = {"P", "map", "map_cut", "ndcg_cut", "Rprec", "recip_rank", "set_recall"}
CUT_POINTS = [5, 10, 15, 20, 30, 100, 200, 500, 1000]


def _mrr(rundoc, qrel):
    """
    calculate the mrr for a list of docs from same query
    :param rundoc: dict, mapping the doc id into doc score
    :param qrel: dict, mapping the doc id into ground truth label
    :return: float, the mrr score
    """
    if (not rundoc) or (not qrel):
        return 0.

    pos_docids, pos_doc_ranks = [d for d in rundoc if qrel.get(d, 0) > 0], []
    if not pos_docids:  # or all([d not in rundoc for d in pos_docids]):
        return 0.

    rundoc = sorted(rundoc.items(), key=lambda doc_score: float(doc_score[1]), reverse=True)
    rundoc = [d for d, i in rundoc]

    pos_doc_ranks = [rundoc.index(d)+1 for d in pos_docids]
    return 1/min(pos_doc_ranks)


def mrr(qrels, runs, qids=None, aggregate=True):
    qids = set(qrels.keys()) & set(runs.keys()) & set(qids) if qids \
        else set(qrels.keys()) & set(runs.keys())

    rundoc_qrel = [(runs.get(q, {}), qrels.get(q, {})) for q in qids]
    with get_context("spawn").Pool(12) as p:
        ranks = p.starmap(_mrr, rundoc_qrel)
    # print("number of runs: ", len(ranks),  sum(ranks) / len(ranks), "number of zero: ", ranks.count(0.0))

    if aggregate:
        return sum(ranks) / len(ranks)
    else:
        return dict(zip(qids, ranks))


def _verify_metric(metrics):
    """
    Verify if the metrics is in the returned list of TREC eval

    Args:
        metrics: a list of str
    """
    assert isinstance(metrics, list)
    expected_metrics = {m for m in VALID_METRICS if not m.endswith("_cut") and m != "P"} | {
        m + "_" + str(cutoff) for cutoff in CUT_POINTS for m in VALID_METRICS if m.endswith("_cut") or m == "P"
    }
    for metric in metrics:
        if metric not in expected_metrics:
            raise ValueError(f"Unexpected evaluation metric: {metric}, should be one of { ' '.join(sorted(expected_metrics))}")


def _transform_metric(metrics):
    """
    Remove the _NUM at the end of metric is applicable

    Args:
        metrics: a list of str

    Returns:
        a set of transformed metric
    """
    assert isinstance(metrics, list)
    metrics = {"_".join(metric.split("_")[:-1]) if "_cut" in metric or "P_" in metric else metric for metric in metrics}
    return metrics


def calc_single_query_runs_trec(qrel, run, metrics):
    trec_metrics = _transform_metric(metrics)
    results = list(pytrec_eval.RelevanceEvaluator(qrel, trec_metrics).evaluate(run).values())[0]   # only one query is supposed to be contained
    results = [results.get(m, -1) for m in metrics]
    return results


def _eval_runs(runs, qrels, metrics, dev_qids):
    assert isinstance(metrics, list)

    calc_mrr = "mrr" in metrics
    if calc_mrr:
        metrics.remove("mrr")
        mrr_score = mrr(qrels, runs, dev_qids)
        scores = {"mrr": mrr_score}
    else:
        scores = {}

    if metrics:  # in case only "mrr" is provided
        _verify_metric(metrics)
        qids = set(dev_qids) & set(qrels.keys()) & set(runs.keys())
        # qrel_run_metrics = [({q: qrels.get(q, {})}, {q: runs.get(q, {})}, metrics) for q in qids]
        # qrel_run_metrics = [
        #     (qrel, run, metrics) for qrel, run, metrics in qrel_run_metrics if len(qrel) > 0 and len(run) > 0]
        # if not qrel_run_metrics:
        #     logger.warning(f"empty qrels runs pair")
        #     return 0
        # print(f"qrel_run_metrics: {len(qrel_run_metrics)}")
        # with get_context("spawn").Pool(12) as p:
        #     trec_scores = p.starmap(calc_single_query_runs_trec, qrel_run_metrics)  # (Q, n_metrics)
        #     assert len(trec_scores) == len(qrel_run_metrics)
        # trec_scores = np.array(trec_scores).mean(axis=0).tolist()
        # if len(trec_scores) != len(metrics):
        #     raise ValueError(f"Mismatch trec_score and metrics, metrics: {metrics}; trec score: {trec_scores}")
        # trec_scores = dict(zip(metrics, trec_scores))
        # scores.update(trec_scores)

        dev_qrels = {qid: labels for qid, labels in qrels.items() if (qid in dev_qids and qid in runs)}
        evaluator = pytrec_eval.RelevanceEvaluator(dev_qrels, _transform_metric(metrics))

        trec_scores = [[metrics_dict.get(m, -1) for m in metrics] for metrics_dict in evaluator.evaluate(runs).values()]
        trec_scores = np.array(trec_scores).mean(axis=0).tolist()
        trec_scores = dict(zip(metrics, trec_scores))
        scores.update(trec_scores)

    return scores


def eval_runs(runs, qrels, metrics):
    """
    Evaluate runs loaded by Searcher.load_trec_run

    Args:
        runs: a dict with format {qid: {docid: score}}, could be prepared by Searcher.load_trec_run
        qrels: dict, containing the judgements provided by benchmark
        metrics: str or list, metrics expected to calculate, e.g. ndcg_cut_20, etc

    Returns:
        a dict with format {metric: score}, containing the evaluation score of specified metrics
    """
    metrics = [metrics] if isinstance(metrics, str) else list(metrics)
    return _eval_runs(runs, qrels, metrics, dev_qids=list(qrels.keys()))


def eval_runfile(runfile, qrels, metrics):
    """
    Evaluate a single runfile produced by ranker or reranker

    Args:
        runfile: str, path to runfile
        qrels: dict, containing the judgements provided by benchmark
        metrics: str or list, metrics expected to calculate, e.g. ndcg_cut_20, etc

    Returns:
        a dict with format {metric: score}, containing the evaluation score of specified metrics
    """
    metrics = [metrics] if isinstance(metrics, str) else list(metrics)
    runs = Searcher.load_trec_run(runfile)
    return _eval_runs(runs, qrels, metrics, dev_qids=list(qrels.keys()))


def search_best_run(runfile_dir, benchmark, primary_metric, metrics=None, folds=None):
    """
    Select the runfile with respect to the specified metric

    Args:
        runfile_dir: the directory path to all the runfiles to select from
        benchmark: Benchmark class
        primary_metric: str, metric used to select the best runfile , e.g. ndcg_cut_20, etc
        metrics: str or list, metric expected by be calculated on the best runs
        folds: str, the name of fold to select from

    Returns:
       a dict storing specified metric score and path to the corresponding runfile
    """
    metrics = [] if not metrics else ([metrics] if isinstance(metrics, str) else list(metrics))
    if primary_metric not in metrics:
        metrics = [primary_metric] + metrics

    folds = {s: benchmark.folds[s] for s in [folds]} if folds else benchmark.folds
    # folds = {s: f for s, f in folds.items() if not os.path.exists(os.path.join(runfile_dir, f".{s}.best"))}
    folds = {s: f for s, f in folds.items() if not glob.glob(os.path.join(runfile_dir, f".{s}.best*"))}
    runfiles = [
        os.path.join(runfile_dir, f)
        for f in os.listdir(runfile_dir)
        if (f != "done" and not os.path.isdir(os.path.join(runfile_dir, f)))
    ]

    # runfiles = ["/home/xinyu1zhang/mpi-spring/capreolus/capreolus/csn_rm3/filtered_bm25/ruby/test.filtered.runfile"]
    # runfiles = ["/home/xinyu1zhang/mpi-spring/capreolus/capreolus/csn_rm3new/filtered_bm25/ruby/test.filtered.runfile"]
    # print("runfiles: ", runfiles)

    best_scores = {s: {primary_metric: 0, "path": None} for s in folds}
    for runfile in tqdm(runfiles, desc="Processing available runfiles"):
        runs = Searcher.load_trec_run(runfile)
        if not runs:
            logger.warning(f"empty file: {runfile}")
            continue

        for s, v in folds.items():
            dev_qids = set(v["train_qids"]) | set(v["predict"]["dev"])
            score = _eval_runs(
                runs, benchmark.qrels, [primary_metric], dev_qids=dev_qids
            )[primary_metric]
            if score > best_scores[s][primary_metric]:
                best_scores[s] = {primary_metric: score, "path": runfile}

    test_runs, test_qrels = {}, {}
    for s, score_dict in best_scores.items():
        best_path = score_dict["path"]
        test_qids = folds[s]["predict"]["test"]
        test_runs.update({qid: v for qid, v in Searcher.load_trec_run(best_path).items() if qid in test_qids})
        test_qrels.update({qid: v for qid, v in benchmark.qrels.items() if qid in test_qids})

        # prepare a copy of the best one
        dirname, basename = os.path.dirname(best_path), os.path.basename(best_path)
        new_file = os.path.join(dirname, f"{s}.best.{basename}")
        with open(new_file, "w") as fout:
            # fout.write(basename+"\n")
            with open(best_path) as f:
                for line in f:
                    fout.write(line)


    scores = eval_runs(test_runs, benchmark.qrels, metrics)
    return {"score": scores, "path": {s: v["path"] for s, v in best_scores.items()}}
