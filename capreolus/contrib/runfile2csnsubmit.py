#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Crystina Zhang time:27/4/2020
import json
import wandb

wandb.init(project="code-search", name="bm25-top100-stem-camel_tok", )

qid2query = json.load(open("/home/xinyu1zhang/mpi-spring/capreolus/capreolus/data/csn_challenge/qidmap.json"))
csnsubmit_file = open("model_predictions.csv", "w")
csnsubmit_file.write("query,language,identifier,url\n")

url2docid_patten = "/home/xinyu1zhang/mpi-spring/capreolus/capreolus/data/csn_corpus/docidmap/%s.json"
# runfile_pattern = "/home/xinyu1zhang/.capreolus/cache/collection-codesearchnet_lang-%s/index-anserini_indexstops-False_stemmer-porter/searcher-BM25_b-0.4_hits-100_k1-0.9/codesearchnet_challenge/searcher"
# runfile_pattern = "/home/xinyu1zhang/.capreolus/cache/collection-codesearchnet_lang-%s/index-anserini_indexstops-False_stemmer-porter/searcher-BM25_b-0.4_hits-500_k1-0.9/codesearchnet_challenge/searcher"
# runfile_pattern = "/home/xinyu1zhang/.capreolus/cache/collection-codesearchnet_camel_parser_lang-%s/index-anserini_indexstops-True_stemmer-none/searcher-BM25_b-0.4_hits-100_k1-0.9/codesearchnet_challenge/searcher"
# runfile_pattern = "/home/xinyu1zhang/.capreolus/cache/collection-codesearchnet_camel_parser_lang-%s/index-anserini_indexstops-False_stemmer-porter/searcher-BM25_b-0.4_hits-100_k1-0.9/codesearchnet_challenge/searcher"
# runfile_pattern = "/home/xinyu1zhang/.capreolus/cache/collection-codesearchnet_camel_parser_lang-%s/index-anserini_indexstops-False_stemmer-porter/searcher-BM25_b-0.75_hits-100_k1-1.2/codesearchnet_challenge/searcher"
# runfile_pattern = "/home/xinyu1zhang/.capreolus/cache/collection-codesearchnet_lang-%s/index-anserini_indexstops-False_stemmer-porter/searcher-BM25_b-0.75_hits-100_k1-1.2/codesearchnet_challenge/searcher"
runfile_pattern = "/home/xinyu1zhang/.capreolus/cache/collection-codesearchnet_camel_parser_lang-%s/index-anserini_indexstops-True_stemmer-none/searcher-BM25_b-0.75_hits-100_k1-1.2/codesearchnet_challenge/searcher"


config = {
    "stem": "none",  # "porter",
    "camel_stem": True,
    "indexstops": True,  # False,
    "k1": 1.2,
    "b": 0.75,
}
wandb.config.update(config)

def get_docid2url_map(lang):
    url2docid = json.load(open(url2docid_patten % lang))
    docid2url = {}
    for url, v in url2docid.items():
        if isinstance(v, list):
            docid = v[0]
            docid2url[docid] = url
        else:
            for doc, docid in v.items():
                docid2url[docid] = url
    return docid2url


for lang in ["ruby", "go", "java", "python", "javascript", "php"]:
    print(f"processing {lang}")
    docid2url = get_docid2url_map(lang)
    with open(runfile_pattern % lang) as f:
        # write query, language, identifier, and url
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            url = docid2url[docid]
            query = qid2query[qid].strip()
            csnsubmit_file.write(f"{query},{lang},_,\"{url}\"\n")


csnsubmit_file.close()
wandb.save("model_predictions.csv")
