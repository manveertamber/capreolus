#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Crystina Zhang time:17/5/2020

import os
import json
import sys
if "/home/xinyu1zhang/mpi-spring/capreolus" not in sys.path:
    sys.path.append("/home/xinyu1zhang/mpi-spring/capreolus")

from argparse import ArgumentParser
import numpy as np

import pytrec_eval

from capreolus.utils.trec import load_qrels
from capreolus.searcher import Searcher
from capreolus.evaluator import mrr

LANGS = ["python", "java", "go", "php", "javascript", "ruby"]


def calc_single_qid_mrr(qrels, runs):
    # qid2score = {}
    qids = set(qrels.keys()) & set(runs.keys())
    qid2score = mrr(qrels, runs, qids, aggregate=False)

    qid2score.update({qid: 0 for qid in runs if qid not in qids})
    qid2score.update({qid: -1 for qid in qrels if qid not in qids})
    # for qid in tqdm(runs, desc="calc mrr for each query"):
    #     if qid not in qids:
    #         qid2score[qid] = 0
    #     else:
    #         qid2score[qid] = mrr({qid: qrels[qid]}, {qid: runs[qid]})

    qid2score = sorted(qid2score.items(), key=lambda kv: kv[1], reverse=True)
    qid2score = {k: v for k, v in qid2score}
    return qid2score


def get_qid2query(qidmap_fn):
    query2qid = json.load(open(qidmap_fn, "r", encoding="utf-8"))
    qid2query = {v: k for k, v in query2qid.items()}
    return qid2query


def write(qid2score, qid2query, outp_fn):
    with open(outp_fn, "w", encoding="utf-8") as fout:
        for i, (qid, score) in enumerate(qid2score.items()):
            rank = i+1
            query = qid2query.get(qid, "")
            fout.write(f"{qid}\t{rank}\t{score}\t{query}\n")
    print(f"finished, write to file: {outp_fn}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--qrel_dir", type=str, required=True)
    parser.add_argument("--runfile_pattern", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, required=True)

    parser.add_argument("--qidmap_dir", type=str, default=None)
    parser.add_argument("--metric", "-m", type=str, default="mrr")
    parser.add_argument("--lang", "-l", type=str, default="all", choices=LANGS+["all"])

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    langs = LANGS if args.lang == "all" else [args.lang]

    join = os.path.join
    for lang in langs:
        print(f"processing lang: {lang}")
        qrels = load_qrels(join(args.qrel_dir, f"{lang}.txt"))
        print(f"\tqrels loaded: {len(qrels)} qrels found in qrels, avg len: {np.mean([len(v) for v in qrels.values()])}")
        ns = args.runfile_pattern.count("%s")
        runs = Searcher.load_trec_run(args.runfile_pattern % ((lang,)*ns))
        print(f"\ttrec run loaded: {len(runs)} qids found in runs, avg len: {np.mean([len(v) for v in runs.values()])}")

        qid2query = get_qid2query(join(args.qidmap_dir, f"{lang}.json")) if args.qidmap_dir else {}
        qid2score = calc_single_qid_mrr(qrels, runs)
        print(f"\tscore calculated: {len(qid2score)} qids found in runs")
        write(qid2score, qid2query, outp_fn=join(args.output_dir, lang))
