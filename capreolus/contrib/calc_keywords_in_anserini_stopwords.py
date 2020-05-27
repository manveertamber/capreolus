#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Crystina Zhang time:26/5/2020

import sys

if "/home/xinyu1zhang/mpi-spring/capreolus" not in sys.path:
    sys.path.append("/home/xinyu1zhang/mpi-spring/capreolus")
from capreolus.utils.common import load_keywords

LANGS = ["python", "java", "go", "php", "javascript", "ruby"]
KEYWORDS = {lang: load_keywords(lang) for lang in LANGS}

from jnius import autoclass
stemmer = "porter"
Analyzer = autoclass("io.anserini.analysis.DefaultEnglishAnalyzer")
analyzer = Analyzer.newStemmingInstance(stemmer)
anserini_tokenize = autoclass("io.anserini.analysis.AnalyzerUtils").analyze

for lang in LANGS:
    print(lang)
    for kw in KEYWORDS[lang]:
        print("\t", kw, anserini_tokenize(kw).toArray())