#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Crystina Zhang time:7/5/2020
import math
from pyserini.index.pyutils import IndexReaderUtils


# index_path = "/home/xinyu1zhang/.capreolus/cache/" \
#              "collection-robust04/index-anserini_tf_indexstops-False_stemmer-porter/index/"
# docid = "FT942-876047504" "Poliomyelitis and Post-Polio"
# total_docno = 528030
# avgdoclen = 330.5510520235593


index_path = "/home/xinyu1zhang/.capreolus/cache/" \
             "collection-codesearchnet_camelstemmer-True_lang-python/index-anserini_tf_indexstops-False_stemmer-porter/index"
total_docno = 460678  # ruby: 164048
docid = "python-FUNCTION-448595"  # "python-FUNCTION-422634" "python-FUNCTION-460229"  # "python-FUNCTION-432881"  # "ruby-FUNCTION-108910"
avgdoclen = 73.8370315057372  # ruby: 50.018622598263924
# query = "read a line of input . prompt and use raw exist to be compatible with other input routines and are ignored . eof error will be raised on eof ."  # "deprecated and not recommended"
# query = "enable or disable a breakpoint given its breakpoint number ."
query = "return the resized filename ( according to width height and filename key ) in the following format : filename - filename key - width x height . ext"
# query = ">>> header factory . header class for version ( 2 . 0 ) traceback ( most recent call last ) : ... pylas . errors . file version not supported : 2 . 0"
utils = IndexReaderUtils(index_path)

docvec = utils.get_document_vector(docid)
doclen = sum(docvec.values())  # 60
print("doclen: ", doclen)

k1, b = 1.2, 0.75


def bm25(tf, idf):
    if math.isnan(tf):
        return math.nan
    numerator = tf
    denominator = tf + (k1 * (1 - b + b * doclen / avgdoclen))
    return idf * numerator / denominator


final_bm25_self = 0
final_bm25_calc = 0
terms = {}
for term in utils.analyze(query):
    # ana_term = utils.analyze(term)
    # ana_term = ana_term[0] if ana_term else term+"unfound"

    tf = docvec.get(term, math.nan)
    df, cf = utils.get_term_counts(term, analyzer=None)
    idf = math.log((total_docno - df + 0.5) / (df + 0.5) + 1)
    selfbm25 = bm25(tf, idf)
    calcbm25 = utils.compute_bm25_term_weight(docid=docid, term=term)

    print(f"{term}", ":",
          "\ttf: ", tf, "idf: ", idf,
          "\t selfbm25:", selfbm25, "\tbm25", calcbm25)
    terms[term] = selfbm25

    if not math.isnan(selfbm25):
        final_bm25_self += selfbm25
    if not math.isnan(calcbm25):
        final_bm25_calc += calcbm25

print("final self: ", final_bm25_self, "\tfinal calc: ", final_bm25_calc)
print(terms)

import numpy as np
print(np.nansum(list(terms.values())))