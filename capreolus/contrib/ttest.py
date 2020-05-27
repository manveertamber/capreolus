#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Crystina Zhang time:24/5/2020


import os
import sys
from pathlib import Path
if "/home/xinyu1zhang/mpi-spring/capreolus" not in sys.path:
    sys.path.append("/home/xinyu1zhang/mpi-spring/capreolus")

from capreolus.evaluator import mrr
from capreolus.utils.trec import load_qrels
from capreolus.searcher import Searcher

import numpy as np
from scipy.stats import ttest_rel

bm25_vs_rm3 = {
    "qrel_path": Path("/home/xinyu1zhang/mpi-spring/capreolus/capreolus/data/csn_corpus/qrels"),  # /with_camelstem-keep_punc-remove_keywords",
    "run_path": Path("/home/xinyu1zhang/.capreolus/cache/aggregated_ictir")  # /with_camelstem-keep_punc-remove_keywords",
}

LANGS = ["ruby", "go", "javascript", "java", "python", "php"]


def sort_dict_value(d):
    kv_dict = sorted(d.items(), key=lambda kv: int(kv[0]))
    # d = {k: v for k, v in kv_dict}
    v = [v for k, v in kv_dict]
    return v


for config in ["with_camelstem-keep_punc-remove_keywords"]:
    print(config)
    for lang in LANGS:
        print(lang)
        q_path = (bm25_vs_rm3["qrel_path"] / config / f"{lang}.txt")
        bm25_r_path = list((bm25_vs_rm3["run_path"] / config / "bm25").glob(f"{lang}-*"))[0]
        rm3_r_path = list((bm25_vs_rm3["run_path"] / config / "rm3").glob(f"{lang}-*"))[0]
        # print(f"\t\tqrel: {q_path}\n\tbm25: {bm25_r_path}\n\trm3: {rm3_r_path}")

        qrels = load_qrels(q_path)
        bm25_runs, rm3_runs = Searcher.load_trec_run(bm25_r_path), Searcher.load_trec_run(rm3_r_path)
        q2s_bm25 = mrr(qrels=qrels, runs=bm25_runs, aggregate=False)
        q2s_rm3 = mrr(qrels=qrels, runs=rm3_runs, aggregate=False)
        # check performance:
        bm25_result, rm3_result = np.mean(list(q2s_bm25.values())).item(), np.mean(list(q2s_rm3.values())).item()

        # find overlap queries
        shared_qids = sorted(list(set(q2s_bm25.keys()) & set(q2s_rm3.keys())))
        q2s_bm25 = {q: s for q, s in q2s_bm25.items() if q in shared_qids}
        q2s_rm3 = {q: s for q, s in q2s_rm3.items() if q in shared_qids}

        # s1, s2 = np.array(list(q2s_bm25.values())), np.array(list(q2s_rm3.values()))
        s1, s2 = sort_dict_value(q2s_bm25), sort_dict_value(q2s_rm3)
        s1, s2 = np.array(s1), np.array(s2)

        t_stat, p_value = ttest_rel(s1, s2)
        print("p value: ", p_value, f"(verify bm25 and rm3: %.4f/%.4f)"%(bm25_result, rm3_result))
