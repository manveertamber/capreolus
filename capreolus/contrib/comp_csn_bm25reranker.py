#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Crystina Zhang time:8/5/2020

# if the neighbour 1k are same:
import json
import math

lang = "python"  # "ruby"
fold = f"/home/xinyu1zhang/mpi-spring/capreolus/capreolus/data/csn_corpus/folds/with_camelstem/{lang}.json"
testqid = json.load(open(fold))["s1"]["predict"]["test"]


def load_runfile(fn):
    print("loading", fn)
    runs = {}
    with open(fn) as f:
        for line in f:
           qid, _, docid, rank, score, _ = line.split()
           rank, score = int(rank), float(score)
           if qid in runs:
                runs[qid][docid] = (docid, rank, score)
           else:
                runs[qid] = {docid: (docid, rank, score)}
        return runs


def compare_neighbour_runfile():
    afn = f"/home/xinyu1zhang/mpi-spring/capreolus/capreolus/csn_runfile_lucenebm25/neighbour1k/{lang}/test.runfile.txt"
    bfn = f"/home/xinyu1zhang/.capreolus/cache/benchmark-codesearchnet_corpus_camelstemmer-True_lang-{lang}/" \
          f"searcher-csn_distractors/codesearchnet_corpus/searcher"
    print("load json")
    a = load_runfile(afn)
    print(bfn)
    b = load_runfile(bfn)

    total = len(testqid)
    mismatch, unfounda, unfoundb = [], [], []

    for q in testqid:
        common = f"{q}\t|"
        lena = len(a[q]) if q in a else -1
        lenb = len(b[q]) if q in b else -1

        if lena == -1:
            unfounda.append(q)
        if lenb == -1:
            # print(f"{common} not found in b")
            unfoundb.append(q)

        if lena < 0 or lenb < 0:
            continue

        if a[q] != b[q]:
            share = set(a[q]) & set(b[q])
            mismatch.append((q, lena, lenb, len(share)))

    # print(mismatch)
    print(unfoundb == unfounda)
    filtered_mismatch = [s for s in mismatch if s[-1] % 1000 != 0]
    print(f"mismatch: {len(mismatch)}", len(filtered_mismatch))
    if filtered_mismatch:
        print(filtered_mismatch[:20])
    print(f"unfound a: {len(unfounda)}", unfounda[:10])
    print(f"unfounf b: {len(unfoundb)}", unfoundb[:10])
    print(f"total : {total}")


def compare_filtered_runfile():
    afn = f"/home/xinyu1zhang/mpi-spring/capreolus/capreolus/csn_runfile_python_nocomma/filtered_bm25/{lang}/test.filtered.runfile"
    bfn = f"/home/xinyu1zhang/.capreolus/cache/collection-codesearchnet_camelstemmer-True_lang-{lang}/" \
          f"index-anserini_tf_indexstops-False_stemmer-porter/" \
          f"benchmark-codesearchnet_corpus_camelstemmer-True_lang-{lang}/searcher-csn_distractors/searcher-BM25_reranker_b-0.75_hits-1000_k1-1.2/codesearchnet_corpus/searcher"

    a = load_runfile(afn)
    b = load_runfile(bfn)
    print("a: ", len(a), "b: ", len(b), list(b.keys())[:10])

    unfounda, unfoundb = set(), set()
    docina_butnot_b = {}
    for q in testqid:
        if q not in a:
            unfounda.add(q)
            if q not in b:
                unfoundb.add(q)
            continue

        docina_butnot_b[q] = {"all": len(a[q]), "unfound": [], "found": []}
        for doc in a[q]:
            if q not in b:
                unfoundb.add(q)
                continue

            if doc in b[q]:
                docina_butnot_b[q]["found"].append((a[q][doc], b[q][doc]))
            else:
                docina_butnot_b[q]["unfound"].append(doc)
                print("unfound")

    print("-" * 10)
    for q in docina_butnot_b:
        if len(docina_butnot_b[q]["unfound"]) != 0:
            print(">>> ", q, docina_butnot_b[q]["all"], len(docina_butnot_b[q]["unfound"]))

        # match = all([(math.fabs(sa[1] - sb[1]) < 0.01) for sa, sb in docina_butnot_b[q]["found"]])
        # if not match:
        not_match = [(sa, sb) for sa, sb in docina_butnot_b[q]["found"] if (math.fabs(sa[-1] - sb[-1]) > 0.6)]
        print(q, len(not_match), "/", len(docina_butnot_b[q]["found"]), not_match)
    print("-"*10)

    print(
        f"total num of runfile: ", len(testqid),
        f"\nunfound a: {len(unfounda)}, unfound b: {len(unfoundb)}; a=b?", unfounda == unfoundb
    )
    unfounda, unfoundb = list(unfounda), list(unfoundb)
    print(unfounda[:10])
    print(unfoundb[:10])


# compare_neighbour_runfile()
compare_filtered_runfile()
