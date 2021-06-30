import math
import os
import subprocess

from capreolus import ConfigOption, constants, get_logger
from capreolus.utils.common import Anserini

from . import Index

logger = get_logger(__name__)  # pylint: disable=invalid-name
MAX_THREADS = constants["MAX_THREADS"]
PACKAGE_PATH = constants["PACKAGE_PATH"]


@Index.register
class MSMarcoDoc_V2_Index(Index):
    module_name = "msdoc_v2"
    config_spec = [
        ConfigOption("indexstops", False, "should stopwords be indexed? (if False, stopwords are removed)"),
        ConfigOption("stemmer", "porter", "stemmer: porter, krovetz, or none"),
    ]

    index_dir = PACKAGE_PATH / "data" / "msdoc_v2" / "indexes"

    def _create_index(self):
        outdir = self.get_index_path()

        # collection_path, document_type, generator_type = self.collection.get_path_and_types()
        if (not self.config["indexstops"]) and (self.config["stemmer"] == "porter"):
            index_name = "msmarco-doc-v2"
        elif (self.config["indexstops"]) and (self.config["stemmer"] is None):
            index_name = "msmarco-doc-v2-keepstopwords-no-stemmer-storecontents"
        else:
            raise ValueError("Unsupported cofiguration")

        src = self.index_dir / index_name
        dst = outdir
        os.makedirs(outdir, exist_ok=True)
        for fn in os.listdir(src):
            os.symlink(src=(src / fn), dst=(dst / fn))


    def get_docs(self, doc_ids):
        return [self.get_doc(doc_id) for doc_id in doc_ids]

    def get_doc(self, docid):
        try:
            if not hasattr(self, "index_utils") or self.index_utils is None:
                self.open()
            return self.index_reader_utils.documentContents(self.reader, self.JString(docid))
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

        JIndexUtils = autoclass("io.anserini.index.IndexUtils")
        JIndexReaderUtils = autoclass("io.anserini.index.IndexReaderUtils")
        self.index_utils = JIndexUtils(index_path)
        self.index_reader_utils = JIndexReaderUtils()

        JFile = autoclass("java.io.File")
        JFSDirectory = autoclass("org.apache.lucene.store.FSDirectory")
        fsdir = JFSDirectory.open(JFile(index_path).toPath())
        self.reader = autoclass("org.apache.lucene.index.DirectoryReader").open(fsdir)
        self.numdocs = self.reader.numDocs()
        self.JTerm = autoclass("org.apache.lucene.index.Term")
        self.JString = autoclass("java.lang.String")
