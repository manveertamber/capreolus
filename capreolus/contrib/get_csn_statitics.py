#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Crystina Zhang time:17/5/2020

import os
import re
import sys
import json
from argparse import ArgumentParser
import numpy as np

if "/home/xinyu1zhang/mpi-spring/capreolus" not in sys.path:
    sys.path.append("/home/xinyu1zhang/mpi-spring/capreolus")
from capreolus.utils.trec import load_qrels
from capreolus.utils.common import load_keywords

LANGS = ["python", "java", "go", "php", "javascript", "ruby"]
KEYWORDS = {lang: load_keywords(lang) for lang in LANGS}

# helper
def topic_gen(topic_fn):
    qid = ""
    with open(topic_fn, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith("<num>"):
                qid = line.replace("<num>", "").replace("Number: ", "").strip()

            if not line.startswith("<title>"):
                continue

            assert qid != ""
            topic = line.replace("<title>", "").strip()
            yield qid, topic.split()
            qid = ""


# helper
def coll_gen(coll_fn):
    docid = ""
    with open(coll_fn, "rt", encoding="utf-8") as f:
        while True:
            line = f.readline().strip()
            if line == "":
                line = f.readline().strip()
                if line == "":
                    break

            if line.startswith("<DOCNO>"):
                docid = line.replace("<DOCNO>", "").replace("</DOCNO>", "").strip()

            if line == "<TEXT>":
                doc = f.readline().strip()
                while True:
                    line = f.readline().strip()
                    if line == "</TEXT>":
                        break
                    doc += line

                assert docid != ""
                yield docid, doc.strip()
                docid = ""


# helper
def calc_length(sent, remove_punc, remove_keywords, lang):
    if remove_keywords:
        sent = sent.split() if isinstance(sent, str) else sent
        sent = [w for w in sent if w.lower() not in KEYWORDS[lang]]

    if remove_punc:
        sent = " ".join(sent) if isinstance(sent, list) else sent
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent).strip()

    if isinstance(sent, str):
        sent = sent.split()

    return len(sent), sent


# helper
def load_id2len_file(fn):
    id2len = {}
    with open(fn) as f:
        for line in f:
            id, l = line.strip().split()
            id2len[id] = int(l)
    return id2len


# helper
def get_qid2query(qidmap_fn):
    query2qid = json.load(open(qidmap_fn, "r", encoding="utf-8"))
    qid2query = {v: k for k, v in query2qid.items()}
    return qid2query


def record_stats(topic_fn, coll_fn, output_fn, remove_punc=False, remove_keywords=False, lang=None):
    topic_outfn, coll_outfn = f"{output_fn}.topic", f"{output_fn}.collection"
    if os.path.exists(topic_outfn) and os.path.exists(coll_outfn):
        # print(f"{topic_outfn} and {coll_outfn} exist, loading from previous files")
        topic_lens, coll_lens = load_id2len_file(topic_outfn), load_id2len_file(coll_outfn)
        return topic_lens, coll_lens

    os.makedirs(os.path.dirname(output_fn), exist_ok=True)

    topic_lens, coll_lens = {}, {}
    empties, empty_fn = [], f"{output_fn}.empties"
    with open(topic_outfn, "w") as fout:
        for i, (qid, query) in enumerate(topic_gen(topic_fn)):
            results = calc_length(query, remove_punc=remove_punc, remove_keywords=False, lang=None)
            qlen, transformed_q = results
            if qlen == 0:
                empties.append((qid, query, transformed_q))
            fout.write(f"{qid}\t{qlen}\n")
            # topic_lens.append(qlen)
            topic_lens[qid] = qlen
    print("\ttotal amount of topic: ", len(topic_lens), end="\t")
    print("avg len: ", np.mean(list(topic_lens.values())),
          "maxlen: ", max(topic_lens.values()), "minlen: ", min(topic_lens.values()),
          f"empty query: {len(empties)} / {len(topic_lens)}")
    if empties:
        with open(empty_fn, "a") as f:
            for qid, q, tq in empties:
                f.write(f"query\t{qid}")
                f.write(f"before\t" + " ".join(q) + "|\nafter\t" + " ".join(tq) + "|\n")

    empties, empty_fn = [], f"{output_fn}.empties"
    with open(coll_outfn, "w") as fout:
        for i, (docid, doc) in enumerate(coll_gen(coll_fn)):
            doclen, transformed_doc = calc_length(doc, remove_punc=remove_punc, remove_keywords=remove_keywords, lang=lang)
            if doclen == 0:
                empties.append((doc, transformed_doc))
            fout.write(f"{docid}\t{doclen}\n")
            # coll_lens.append(doclen)
            coll_lens[docid] = doclen
    print("\ttotal amount of collection: ", len(coll_lens), end="\t")
    print("avg len: ", np.mean(list(coll_lens.values())), f"empty doc: {len(empties)} / {len(coll_lens)}")
    if empties:
        with open(empty_fn, "a") as f:
            for doc, tdoc in empties:
                f.write(f"query\tbefore\t" + " ".join(doc) + "|\n")
                f.write(f"query\tafter\t" + " ".join(tdoc) + "|\n")

    return topic_lens, coll_lens


def split_qid_by_set(qid2len, folds):
    train, valid, test = folds["train_qids"], folds["predict"]["dev"], folds["predict"]["test"]
    # set2len = {}
    for set_name, qids in zip(["train", "valid", "test"], [train, valid, test]):
        # set2len[set_name] = {q: qid2len[q] for q in qids}
        set2len = {q: qid2len[q] for q in qids}
        unfounds = [l for l in set2len.values() if l < 0]
        empty = [l for l in set2len.values() if l == 0]
        smaller_then_three = [l for l in set2len.values() if 0 < l < 3]
        print(f"\t{set_name}",
              f"\tquery no: {len(set2len)}"
              f"\tavg len: {np.mean(list(set2len.values()))}"
              f"\tunfound: {len(unfounds)}\tempty: {len(empty)}\t<3 token: {len(smaller_then_three)}")


def inspect_dup_query(qrels, folds, qid2query):
    nqids = len(qrels)
    ntotal = sum([len(d) for q, d in qrels.items()])
    print(f"\tquery len: {nqids}\ttotal len: {ntotal}")


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--porter", "-p", type=bool, required=True)
    # parser.add_argument("--stopwords", )
    # parser.add_argument("--camel", "-c", type=bool, required=True)  # embedded with collection and benchmark

    # path
    parser.add_argument("--collection_fn", "-c", type=str)
    parser.add_argument("--topic_fn", "-t", type=str)
    parser.add_argument("--folds", type=str, default=None)
    parser.add_argument("--qrels", type=str, default=None)
    parser.add_argument("--qidmap_dir", type=str, default=None)

    parser.add_argument("--output_dir", "-o", type=str)

    # config
    parser.add_argument("--lang", "-l", type=str, default="all", choices=LANGS+["all"])
    parser.add_argument("--remove_punc", "-punc", type=bool, default=False)
    parser.add_argument("--split_camel", "-camel", type=bool, default=False)
    parser.add_argument("--remove_keywords", "-key", type=bool, default=False)

    args = parser.parse_args()

    # main
    os.makedirs(args.output_dir, exist_ok=True)
    langs = LANGS if args.lang == "all" else [args.lang]

    punc_str = "remove_punc" if args.remove_punc else "keep_punc"
    keyw_str = "remove_keywords" if args.remove_keywords else "keep_keywords"
    for lang in langs:
        print(f"processing language: {lang}")
        outp_fn = os.path.join(args.output_dir, f"{punc_str}-{keyw_str}", f"{lang}")
        qid2len, docid2len = record_stats(
            topic_fn=os.path.join(args.topic_fn, f"{lang}.txt"),
            coll_fn=os.path.join(f"{args.collection_fn}-{lang}", "documents", f"csn-{lang}-collection.txt"),
            output_fn=outp_fn,
            remove_punc=args.remove_punc,
        )
        if args.folds:
            folds = json.load(open(os.path.join(args.folds, f"{lang}.json")))["s1"]
            split_qid_by_set(qid2len, folds)
        if args.qrels:
            qrels = load_qrels(os.path.join(args.qrels, f"{lang}.txt"))
            inspect_dup_query(qrels, None, None)