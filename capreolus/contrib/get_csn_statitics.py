#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Crystina Zhang time:17/5/2020

import os
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


def calc_length(sent, parser):
    if isinstance(sent, str):
        sent = sent.split()
    return len(sent)


def record_stats(topic_fn, coll_fn, output_fn):
    topic_lens, coll_lens = [], []
    with open(f"{output_fn}.topic", "w") as fout:
        for i, (qid, query) in enumerate(topic_gen(topic_fn)):
            # if i % 50000 == 0:
            #     print(f"\t\t{i} topics done")
            qlen = calc_length(query, None)
            fout.write(f"{qid}\t{qlen}\n")
            topic_lens.append(qlen)
    print("\ttotal amount of topic: ", len(topic_lens), end="\t")
    print("avg len: ", np.mean(topic_lens))

    with open(f"{output_fn}.collection", "w") as fout:
        for i, (docid, doc) in enumerate(coll_gen(coll_fn)):
            # if i % 500000 == 0:
            #     print(f"\t\t{i} docs done")
            doclen = calc_length(doc, None)
            fout.write(f"{docid}\t{doclen}\n")
            coll_lens.append(doclen)
    print("\ttotal amount of collection: ", len(coll_lens), end="\t")
    print("avg len: ", np.mean(coll_lens))


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--porter", "-p", type=bool, required=True)
    # parser.add_argument("--stopwords", )
    # parser.add_argument("--camel", "-c", type=bool, required=True)  # embedded with collection and benchmark

    parser.add_argument("--collection_fn", "-c", type=str)
    parser.add_argument("--topic_fn", "-t", type=str)
    parser.add_argument("--output_dir", "-o", type=str)
    parser.add_argument("--lang", "-l", type=str, default="all", choices=LANGS+["all"])

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    langs = LANGS if args.lang == "all" else [args.lang]

    for lang in langs:
        print(f"processing language: {lang}")
        outp_fn = os.path.join(args.output_dir, lang)
        record_stats(
            topic_fn=os.path.join(args.topic_fn, f"{lang}.txt"),
            output_fn=outp_fn,
            coll_fn=os.path.join(f"{args.collection_fn}-{lang}", "documents", f"csn-{lang}-collection.txt"))
