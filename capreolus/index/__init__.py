import logging
import math
import os
import json
import subprocess
from itertools import islice
from threading import Thread
from multiprocessing import get_context, Process

import numpy as np
from tqdm import tqdm
from pyserini.index.pyutils import IndexReaderUtils

from capreolus.registry import ModuleBase, RegisterableModule, Dependency, MAX_THREADS
from capreolus.utils.common import Anserini
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


class Index(ModuleBase, metaclass=RegisterableModule):
    """the module base class"""

    module_type = "index"
    dependencies = {"collection": Dependency(module="collection")}

    def get_index_path(self):
        return self.get_cache_path() / "index"

    def exists(self):
        donefn = self.get_index_path() / "done"
        return donefn.exists()

    def create_index(self):
        if self.exists():
            return

        self._create_index()
        donefn = self.get_index_path() / "done"
        with open(donefn, "wt") as donef:
            print("done", file=donef)

    def _create_index(self):
        raise NotImplementedError()

    def get_doc(self, doc_id):
        raise NotImplementedError()

    def get_docs(self, doc_ids):
        raise NotImplementedError()


def get_cache_path(self):
    print(self.get_cache_path())


class AnseriniIndex(Index):
    name = "anserini"
    commands = {"cache_path": get_cache_path}

    @staticmethod
    def config():
        indexstops = False
        stemmer = "porter"

    def _create_index(self):
        outdir = self.get_index_path()
        stops = "-keepStopwords" if self.cfg["indexstops"] else ""

        collection_path, document_type, generator_type = self["collection"].get_path_and_types()

        anserini_fat_jar = Anserini.get_fat_jar()
        if self["collection"].is_large_collection:
            cmd = f"java -classpath {anserini_fat_jar} -Xms512M -Xmx31G -Dapp.name='IndexCollection' io.anserini.index.IndexCollection -collection {document_type} -generator {generator_type} -threads {MAX_THREADS} -input {collection_path} -index {outdir} -stemmer {self.cfg['stemmer']} {stops}"
        else:
            cmd = f"java -classpath {anserini_fat_jar} -Xms512M -Xmx31G -Dapp.name='IndexCollection' io.anserini.index.IndexCollection -collection {document_type} -generator {generator_type} -threads {MAX_THREADS} -input {collection_path} -index {outdir} -storePositions -storeDocvectors -storeContents -stemmer {self.cfg['stemmer']} {stops}"

        logger.info("building index %s", outdir)
        logger.debug(cmd)
        os.makedirs(os.path.basename(outdir), exist_ok=True)

        app = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)

        # Anserini output is verbose, so ignore DEBUG log lines and send other output through our logger
        for line in app.stdout:
            Anserini.filter_and_log_anserini_output(line, logger)

        app.wait()
        if app.returncode != 0:
            raise RuntimeError("command failed")

    def get_docs(self, doc_ids):
        # if self.collection.is_large_collection:
        #     return self.get_documents_from_disk(doc_ids)
        return [self.get_doc(doc_id) for doc_id in doc_ids]

    def get_doc(self, docid):
        try:
            if not hasattr(self, "index_utils") or self.index_utils is None:
                self.open()
            return self.index_reader_utils.doc_contents(docid)
        except Exception as e:
            raise

    def get_df(self, term):
        # returns 0 for missing terms
        if not hasattr(self, "reader") or self.reader is None:
            self.open()
        jterm = self.JTerm("contents", term)
        return self.reader.docFreq(jterm)

    def get_idf(self, term):
        """ BM25's IDF with a floor of 0 """
        df = self.get_df(term)
        idf = (self.numdocs - df + 0.5) / (df + 0.5)
        idf = math.log(1 + idf)
        return max(idf, 0)

    def open(self):
        from jnius import autoclass

        index_path = self.get_index_path().as_posix()
        self.index_path = index_path

        JIndexUtils = autoclass("io.anserini.index.IndexUtils")
        self.index_utils = JIndexUtils(index_path)

        JFile = autoclass("java.io.File")
        JFSDirectory = autoclass("org.apache.lucene.store.FSDirectory")
        fsdir = JFSDirectory.open(JFile(index_path).toPath())
        self.reader = autoclass("org.apache.lucene.index.DirectoryReader").open(fsdir)
        self.numdocs = self.reader.numDocs()
        self.JTerm = autoclass("org.apache.lucene.index.Term")

        self.index_reader_utils = IndexReaderUtils(index_path)


class AnseriniIndexWithTf(AnseriniIndex):
    name = "anserini_tf"
    commands = {"cache_path": get_cache_path}

    @staticmethod
    def config():
        indexstops = False
        stemmer = "porter"

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
        return self.index_reader_utils.get_document_vector(docid)

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

        # if not term or term in ["in", "at", "an", "it", "on", "be", "their", "or", "is", "no", "with", "will"]:
        #     return 0

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
            return math.nan

        if docid not in self.tf:
            docvec = self.get_doc_vec(docid)  # a dict that map docid to docvec, empty if not found
            self.tf[docid] = docvec

        return self.tf[docid].get(term, math.nan)

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
            docnos = self["collection"].get_docnos()
            for docid in tqdm(docnos, desc="Preparing doclen & tf"):
                self.doclen[docid], self.tf[docid] = self.calc_doclen_tfdict(docid)

            json.dump(self.tf, open(tf_path, "w"))
            print(f"{len(self.tf)} tf values been cached into {tf_path}")

        self.avgdl = np.mean(list(self.doclen.values()))
        print(f"average doc len: {self.avgdl}")
