#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Crystina Zhang time:17/5/2020

import os
import re
from argparse import ArgumentParser
import numpy as np

LANGS = ["python", "java", "go", "php", "javascript", "ruby"]


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


def calc_length(sent, remove_punc):
    if remove_punc:
        sent = " ".join(sent) if isinstance(sent, list) else sent
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent).strip()

    if isinstance(sent, str):
        sent = sent.split()

    return len(sent), sent


def record_stats(topic_fn, coll_fn, output_fn, remove_punc=False):
    os.makedirs(os.path.dirname(output_fn), exist_ok=True)

    topic_lens, coll_lens = [], []
    empties, empty_fn = [], f"{output_fn}.empties"
    with open(f"{output_fn}.topic", "w") as fout:
        for i, (qid, query) in enumerate(topic_gen(topic_fn)):
            results = calc_length(query, remove_punc=remove_punc)
            qlen, transformed_q = results
            if qlen == 0:
                empties.append((qid, query, transformed_q))
            fout.write(f"{qid}\t{qlen}\n")
            topic_lens.append(qlen)
    print("\ttotal amount of topic: ", len(topic_lens), end="\t")
    print("avg len: ", np.mean(topic_lens), f"empty query: {topic_lens.count(0)} / {len(topic_lens)}")
    if empties:
        with open(empty_fn, "a") as f:
            for qid, q, tq in empties:
                f.write(f"query\t{qid}")
                f.write(f"before\t" + " ".join(q) + "|\nafter\t" + " ".join(tq) + "|\n")

    empties, empty_fn = [], f"{output_fn}.empties"
    with open(f"{output_fn}.collection", "w") as fout:
        for i, (docid, doc) in enumerate(coll_gen(coll_fn)):
            doclen, transformed_doc = calc_length(doc, remove_punc=remove_punc)
            if doclen == 0:
                empties.append((doc, transformed_doc))
            fout.write(f"{docid}\t{doclen}\n")
            coll_lens.append(doclen)
    print("\ttotal amount of collection: ", len(coll_lens), end="\t")
    print("avg len: ", np.mean(coll_lens), f"empty doc: {coll_lens.count(0)} / {len(coll_lens)}")
    if empties:
        with open(empty_fn, "a") as f:
            for doc, tdoc in empties:
                f.write(f"query\tbefore\t" + " ".join(doc) + "|\n")
                f.write(f"query\tafter\t" + " ".join(tdoc) + "|\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--porter", "-p", type=bool, required=True)
    # parser.add_argument("--stopwords", )
    # parser.add_argument("--camel", "-c", type=bool, required=True)  # embedded with collection and benchmark

    parser.add_argument("--collection_fn", "-c", type=str)
    parser.add_argument("--topic_fn", "-t", type=str)
    parser.add_argument("--output_dir", "-o", type=str)
    parser.add_argument("--lang", "-l", type=str, default="all", choices=LANGS+["all"])

    parser.add_argument("--remove_punc", "-punc", type=bool, default=False)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    langs = LANGS if args.lang == "all" else [args.lang]

    punc_str = "remove_punc" if args.remove_punc else "keep_punc"
    for lang in langs:
        print(f"processing language: {lang}")
        outp_fn = os.path.join(args.output_dir, punc_str, f"{lang}")
        record_stats(
            topic_fn=os.path.join(args.topic_fn, f"{lang}.txt"),
            coll_fn=os.path.join(f"{args.collection_fn}-{lang}", "documents", f"csn-{lang}-collection.txt"),
            output_fn=outp_fn,
            remove_punc=args.remove_punc,
        )
