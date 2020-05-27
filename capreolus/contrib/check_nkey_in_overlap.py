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
from capreolus.utils.common import load_keywords, remove_newline, get_code_parser

LANGS = ["python", "java", "go", "php", "javascript", "ruby"]
KEYWORDS = {lang: load_keywords(lang) for lang in LANGS}

# p = "/home/xinyu1zhang/mpi-spring/capreolus/capreolus/csn_final_stat/remove_punc-not_split_camel-keep_keywords"
p = "/home/xinyu1zhang/mpi-spring/capreolus/capreolus/csn_final_stat/remove_punc-not_split_camel-remove_keywords"
for lang in LANGS:
    overjson_fn = f"{p}/{lang}/vocab.overlap.json"
    overl = json.load(open(overjson_fn))
    vocabs = set(overl.values())
    kw = KEYWORDS[lang]
    n_in = [w for w in kw if w in vocabs]
    rate = len(n_in) / len(kw)
    print(f"{lang}: num found: {len(n_in)}; num kw loaded: {len(kw)}; num total vocab: {len(vocabs)}; ratio: {rate}")
