import os
import math
import json
import jnius

import numpy as np
from tqdm import tqdm

from capreolus import ConfigOption, constants, get_logger

from . import Index
from .anserini import AnseriniIndex


@Index.register
class AnseriniIndexWithTf(AnseriniIndex):
    module_name = "anserini_tf"

    config_spec = [
        ConfigOption("indexstops", False, "should stopwords be indexed? (if False, stopwords are removed)"),
        ConfigOption("stemmer", "porter", "stemmer: porter, krovetz, or none"),
    ]

    def analyze_term(self, term):
        self.open()
        if term in self.term_analyze:
            return self.term_analyze[term]

        analyzed_term = self.index_reader_utils.analyze(term) if self.cfg["stemmer"] == "porter" else [term]
        analyzed_term = analyzed_term[0] if analyzed_term else None
        self.term_analyze[term] = analyzed_term
        return analyzed_term

    def analyze_sent(self, sent):
        return self.index_reader_utils.analyze(sent)

    def get_doc_vec(self, docid):
        self.open()
        if isinstance(docid, int):
            docid = self.index_reader_utils.convert_internal_docid_to_collection_docid(docid)
        try:
            docvec = self.index_reader_utils.get_document_vector(docid)
        except jnius.JavaException as e:
            print("docid: ", docid, e)
            docvec = {}
        return docvec

    def calc_doclen_tfdict(self, docid):
        doc_vec = self.get_doc_vec(docid)
        doc_vec = {k: v for k, v in doc_vec.items() if v}
        return sum(doc_vec.values()), doc_vec

    def get_doclen(self, docid):
        self.open()
        doclen = self.doclen.get(docid, -1)
        return doclen if doclen != 0 else -1

    def get_avglen(self):
        self.open()
        return self.avgdl

    def get_df(self, term, analyze=False):
        if analyze:
            term = self.analyze(term)

        try:
            df, _ = self.index_reader_utils.get_term_counts(term, analyzer=None)
            return df
        except Exception as e:
            print(term, "|", e)
            return 0

    def get_idf(self, term):
        """ BM25's IDF with a floor of 0 """
        self.open()
        if term in self.idf:
            return self.idf[term]

        df = self.get_df(term)
        idf = (self.numdocs - df + 0.5) / (df + 0.5)
        idf = math.log(1 + idf)
        self.idf[term] = idf

        return idf

    def get_tf(self, term, docid, analyze=False):
        """
        :param term: term in unanalyzed form
        :param docid: the collection document id
        :return: float, the tf of the term if the term is found, otherwise math.nan
        """
        if analyze:
            term = self.analyze(term)
        if not term:
            return 0
        return self.tf[docid].get(term, 0.0)

    def get_bm25_weight(self, term, docid, analyze=False):
        """
        :param term: term in unanalyzed form ?? # TODO, CONFORM
        :param docid: the collection document id
        :return: float, the tf of the term if the term is found, otherwise math.nan
        """
        if analyze:
            term = self.analyze(term)
        return self.index_reader_utils.compute_bm25_term_weight(docid, term) if term else math.nan

    def open(self):
        if hasattr(self, "tf"):
            assert all([hasattr(self, f) for f in ["idf", "doclen", "term_analyze"]])
            return

        if not hasattr(self, "index_utils"):
            super().open()

        self.tf, self.idf, self.doclen, self.term_analyze = {}, {}, {}, {}

        cache_path = self.get_cache_path()
        cache_path.mkdir(exist_ok=True, parents=True)
        tf_path = self.get_cache_path() / "tf.json"
        if os.path.exists(tf_path):
            self.tf = json.load(open(tf_path))
            self.doclen = {docid: sum(doc_vec.values()) for docid, doc_vec in self.tf.items()}
            print(f"tf been loaded from {tf_path}")
        else:
            docnos = self.collection.get_docnos()
            for docid in tqdm(docnos, desc="Preparing doclen & tf"):
                self.doclen[docid], self.tf[docid] = self.calc_doclen_tfdict(docid)

            json.dump(self.tf, open(tf_path, "w"))
            print(f"{len(self.tf)} tf values been cached into {tf_path}")

        self.avgdl = np.mean(list(self.doclen.values()))
        print(f"average doc len: {self.avgdl}")