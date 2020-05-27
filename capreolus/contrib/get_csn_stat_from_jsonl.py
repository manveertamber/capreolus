#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Crystina Zhang time:17/5/2020

import os
import re
import sys
import gzip
import json
from pathlib import Path
from argparse import ArgumentParser
import numpy as np

if "/home/xinyu1zhang/mpi-spring/capreolus" not in sys.path:
    sys.path.append("/home/xinyu1zhang/mpi-spring/capreolus")
from capreolus.utils.trec import load_qrels
from capreolus.utils.common import load_keywords, remove_newline, get_code_parser

# LANGS = ["python", "java", "go", "php", "javascript", "ruby"]
LANGS = ["ruby", "javascript", "go", "python", "java", "php"]
KEYWORDS = {lang: load_keywords(lang) for lang in LANGS}

# added
# import jnius_config
# path_to_anserini_fat_jar = ""  # better to be previous release version? the latest one updated a ton of api
# jnius_config.set_classpath(path_to_anserini_fat_jar)

# after config
from jnius import autoclass
stemmer = "porter"

Analyzer = autoclass("io.anserini.analysis.DefaultEnglishAnalyzer")
analyzer = Analyzer.newStemmingInstance(stemmer)
anserini_tokenize = autoclass("io.anserini.analysis.AnalyzerUtils").analyze
# end of


# helper
def gen_doc_from_gzdir(dir):
    """ generate parsed dict-format doc from all jsonl.gz files under given directory """
    for fn in sorted(dir.glob("*.jsonl.gz")):
        f = gzip.open(fn, "rb")
        for doc in f:
            yield json.loads(doc)


# helper
def process_sentence(sent, split_camel, remove_keywords, lang, anserini_tok):
    sent = re.sub('[^A-Za-z_]+', ' ', sent).strip()  # always remove non-alphabatic
    sent = " ".join(sent) if isinstance(sent, list) else sent

    if split_camel:  # will handle remove_keywords internally
        parser = get_code_parser()

        kws = KEYWORDS[lang] if remove_keywords else []
        sent = parser(sent, kws)  # expect string

    elif remove_keywords:
        sent = re.sub('[^A-Za-z_]+', ' ', sent).strip()
        sent = sent.split() if isinstance(sent, str) else sent
        sent = [w for w in sent if w.lower() not in KEYWORDS[lang]]

    if anserini_tok:
        sent = " ".join(sent) if isinstance(sent, list) else sent
        sent = anserini_tokenize(analyzer, sent).toArray()

    if isinstance(sent, str):
        sent = sent.split()

    return sent


def get_dict(lang, s):
    return {f"{lang}-{i}": w for i, w in enumerate(s)}


def prep_parse_topic_file(data_dir, outp_dir, langs, remove_punc, split_camel, remove_keywords, anserini_tok):
    for lang in langs:
        print(f"processing langugage: {lang}")
        dirname = os.path.join(outp_dir, lang)
        os.makedirs(dirname, exist_ok=True)

        kw2df = {kw: 0 for kw in KEYWORDS[lang]}
        kw2querydf = {kw: [] for kw in KEYWORDS[lang]}

        qid = 0
        n_overlong_docstring = 0
        qvocab_all, doc_vocab_all = set(), set()
        qlen_all, doclen_all, overlap_all = [], [], []
        for set_name in ["train", "valid", "test"]:
            qlens, doclens = [], []
            q_vocabs, doc_vocabs = set(), set()  # write to json file?
            n_overlap = []

            lenfile = open(os.path.join(dirname, f"{set_name}.lenfile"), "w")

            set_path = data_dir / lang / "final" / "jsonl" / set_name

            for doc in gen_doc_from_gzdir(set_path):
                url = doc["url"]
                code, docstring = " ".join(doc["code_tokens"]), " ".join(doc["docstring_tokens"])

                # process_sentence returns a list of tokens
                docstring = \
                    process_sentence(docstring, split_camel, remove_keywords=False, lang=lang, anserini_tok=anserini_tok)
                code = \
                    process_sentence(code, split_camel, remove_keywords=remove_keywords, lang=lang, anserini_tok=anserini_tok)
                n_same_toks = [w for w in docstring if w in code]

                # update keyword df:
                for kw in kw2df:
                    if kw in code:
                        kw2df[kw] += 1
                    if kw in docstring:
                        kw2querydf[kw].append(docstring.count(kw))

                # record vocab
                q_vocabs.update(docstring)
                doc_vocabs.update(code)

                # record length, overlap (between matched query and doc)
                nq, nd, no = len(docstring), len(code), len(n_same_toks)
                qlens.append(nq)
                doclens.append(nd)
                n_overlap.append(no)

                # record length into file (qid, qlen, doclen)
                qid_str = f"{lang}-{qid}"
                lenfile.write(f"{qid_str}\t{url}\t{nq}\t{nd}\t{no}\n")
                qid += 1

            total_qd = q_vocabs | doc_vocabs
            overlap_qd = q_vocabs & doc_vocabs
            print(f"{lang}\t{set_name}\t"
                  f"avgqlen: {np.mean(qlens)}\t"
                  f"avg doclen: {np.mean(doclens)}\t"
                  f"avg qd overlap: {np.mean(n_overlap)}\t"
                  f"\n\tq vocab: {len(q_vocabs)}; doc vocab: {len(doc_vocabs)}; "
                  f"overlap: {len(overlap_qd)}; all: {len(total_qd)}",
                  f"overlap_ratio: %.4f" % (len(overlap_qd) / len(total_qd)))

            qvocab_all.update(q_vocabs)
            doc_vocab_all.update(doc_vocabs)
            qlen_all.extend(qlens)
            doclen_all.extend(doclens)
            overlap_all.extend(n_overlap)

        total_qd = qvocab_all | doc_vocab_all
        overlap_qd = qvocab_all & doc_vocab_all

        kw2df["total_data_num"] = qid

        json.dump(get_dict(lang, overlap_qd), open(f"{dirname}/vocab.overlap.json", "w", encoding="utf-8"))
        json.dump(get_dict(lang, total_qd), open(f"{dirname}/vocab.total.json", "w", encoding="utf-8"))
        json.dump(kw2df, open(f"{dirname}/keywords2df.json", "w"))
        json.dump(kw2querydf, open(f"{dirname}/keywords2querydf.json", "w"))

        print(f"{lang}\tALL"
              f"avgqlen: {np.mean(qlen_all)}\t"
              f"avg doclen: {np.mean(doclen_all)}\t"
              f"avg qd overlap: {np.mean(overlap_all)}\t"
              f"\tq vocab: {len(qvocab_all)}; doc vocab: {len(doc_vocab_all)}; "
              f"overlap: {len(overlap_qd)}; all: {len(total_qd)}, "
              f"overlap_ratio: %.4f" % (len(overlap_qd) / len(total_qd)))
        print()


if __name__ == "__main__":
    parser = ArgumentParser()
    # path
    # parser.add_argument("--collection_fn", "-c", type=str)
    # parser.add_argument("--topic_fn", "-t", type=str)
    # parser.add_argument("--folds", type=str, default=None)
    # parser.add_argument("--qrels", type=str, default=None)
    # parser.add_argument("--qidmap_dir", type=str, default=None)
    parser.add_argument("--output_dir", "-o", type=str, required=True)

    parser.add_argument("--raw_file_dir", type=str, default="/tmp/")

    # config
    parser.add_argument("--lang", "-l", type=str, default="all", choices=LANGS+["all"])
    parser.add_argument("--remove_punc", "-punc", type=bool, default=False)
    parser.add_argument("--split_camel", "-camel", type=bool, default=False)
    parser.add_argument("--remove_keywords", "-key", type=bool, default=False)
    parser.add_argument("--anserini_tok", "-ans", type=bool, default=False)

    args = parser.parse_args()

    # main
    os.makedirs(args.output_dir, exist_ok=True)
    langs = LANGS if args.lang == "all" else [args.lang]

    punc_str = "remove_punc" if args.remove_punc else "keep_punc"
    split_camel = "split_camel" if args.split_camel else "not_split_camel"
    keyw_str = "remove_keywords" if args.remove_keywords else "keep_keywords"
    ans_str = "anserini_tok" if args.anserini_tok else "no_anserini_tok"

    outp_dir = os.path.join(args.output_dir, f"{punc_str}-{split_camel}-{keyw_str}-{ans_str}")
    outp_dir = Path(outp_dir)
    prep_parse_topic_file(
        data_dir=Path(args.raw_file_dir),
        outp_dir=outp_dir,
        langs=langs,
        remove_punc=args.remove_punc,  # not used
        split_camel=args.split_camel,
        remove_keywords=args.remove_keywords,
        anserini_tok=args.anserini_tok,
    )

