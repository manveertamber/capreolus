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
    """This index read documents from msmarco doc v2 colleciton or presegmented doc v2 collection"""

    module_name = "msdoc_v2"
    config_spec = [
        ConfigOption("fields", ["url", "title", "headings", "body"], "which fields to include", value_type="strlist"),
    ]

    def _create_index(self):
        collection = self.collection.module_name
        supported_collections = ["msdoc_v2", "msdoc_v2_preseg"]
        if collection not in supported_collections:
            raise ValueError(f"Not supported collection module: {collection}, should be one of {supported_collections}.")

    def get_docs(self, doc_ids):
        return [self.get_doc(doc_id) for doc_id in doc_ids]

    def get_doc(self, docid):
        doc = self.collection.get_doc()  # dictionary format
        return " ".join([doc.get(field, "") for field in self.config["fields"]])

    def get_passages(self, docid):
        # todo: allow msdoc_v2 also able to call this function 
        assert "#" not in docid
        assert self.colleciton.module_name == "msdoc_v2_preseg"
        passages = self.collection.get_passages(docid)  # dictionary format
        return [" ".join([passage.get(field, "") for field in self.config["fields"]]) for passage in passages]

    def get_df(self, term):
        return None

    def get_idf(self, term):
        return None


@Index.register
class MsPsgV2Index(Index):
    """This index read documents from msmarco doc v2 colleciton or presegmented doc v2 collection"""

    module_name = "mspsg_v2"
    dependencies = [
        Dependency(key="collection", module="collection", name="mspsg_v2"),
    ]

    def _create_index(self):
        collection = self.collection.module_name
        if collection != "msdoc_v2_preseg":
            raise ValueError(f"Not supported collection module: {collection}, should msdoc_v2_preseg.")

    def get_docs(self, doc_ids):
        return [self.get_doc(doc_id) for doc_id in doc_ids]

    def get_doc(self, docid):
        self.colleciton.get_doc(docid)

    def get_df(self, term):
        return None

    def get_idf(self, term):
        return None
