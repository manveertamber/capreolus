#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Crystina Zhang time:16/5/2020

p = "/home/xinyu1zhang/mpi-spring/capreolus/capreolus/data/qrels.robust2004.txt"

from collections import defaultdict

def load_q(fn):
    qrels = defaultdict(dict)
    with open(fn) as f:
        for l in f:
            q, _, d, l = l.strip().split()
            qrels[q][d] = int(l)
    return qrels


def keep_topk(qrels, k):
    for q in qrels:
        docs = qrels[q]
        if len(docs) < k:
            continue

        docs = sorted(docs.items(), key=lambda kv:kv[1], reverse=True)
        qrels[q] = {k: v for k, v in docs[:k]}
    return qrels


def write_qrels(qrels, fn):
    with open(fn, "w") as f:
        for q, d_label in qrels.items():
            for d, l in d_label.items():
                f.write(f"{q} Q0 {d} {l}\n")
    print("finished")


qrels = load_q(p)
qrels = keep_topk(qrels)
write_qrels(q, "after_filter.qrel")