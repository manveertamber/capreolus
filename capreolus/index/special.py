import math
import os
import subprocess

from capreolus import Dependency, ConfigOption, constants, get_logger
from capreolus.utils.common import Anserini

from . import Index

logger = get_logger(__name__)  # pylint: disable=invalid-name
MAX_THREADS = constants["MAX_THREADS"]


@Index.register
class MsV2Index(Index):
    """This index read documents from msmarco doc v2 collection or presegmented doc v2 collection"""

    module_name = "msdocv2index"

    def _create_index(self):
        collection = self.collection.module_name
        supported_collections = ["msdoc_v2", "msdoc_v2_preseg"]
        if collection not in supported_collections:
            raise ValueError(f"Not supported collection module: {collection}, should be one of {supported_collections}.")
        outdir = self.get_index_path()
        os.makedirs(outdir, exist_ok=True)

    def get_docs(self, doc_ids):
        return [self.get_doc(doc_id) for doc_id in doc_ids]

    def get_doc(self, docid):
        doc = self.collection.get_doc(docid)  # dictionary format
        return " ".join([doc.get(field, "") for field in self.config["fields"]])

    def get_passages(self, docid):
        return self.collection.get_passages(docid)  # dictionary format

    def get_df(self, term):
        return None

    def get_idf(self, term):
        return None


@Index.register
class MsPsgV2Index(Index):
    """This index read documents from msmarco doc v2 collection or presegmented doc v2 collection"""

    module_name = "mspsgv2index"
    dependencies = [
        Dependency(key="collection", module="collection", name="mspsg_v2"),
    ]

    def _create_index(self):
        collection = self.collection.module_name
        if collection != "mspsg_v2":
            raise ValueError(f"Not supported collection module: {collection}, should mspsg_v2.")
        outdir = self.get_index_path()
        os.makedirs(outdir, exist_ok=True)

    def get_docs(self, doc_ids):
        return [self.get_doc(doc_id) for doc_id in doc_ids]

    def get_doc(self, docid):
        return self.collection.get_doc(docid)

    def get_passages(self, docid):
        return [self.get_doc(docid)]

    def get_df(self, term):
        return None

    def get_idf(self, term):
        return None
