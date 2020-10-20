import math
import os
import subprocess

from capreolus import ConfigOption, Dependency, constants, get_logger
from capreolus.utils.common import Anserini

from . import Index

logger = get_logger(__name__)  # pylint: disable=invalid-name


@Index.register
class MSMARCOIndex(Index):
    module_name = "msmarco_index"
    dependencies = [
        Dependency(key="collection", module="collection", name="msmarcopsg")
    ]
    config_spec = [
        ConfigOption("indexstops", False, "should stopwords be indexed? (if False, stopwords are removed)"),
        ConfigOption("stemmer", "porter", "stemmer: porter, krovetz, or none"),
    ]

    def exists(self):
        return True

    def get_doc(self, docid):
        return self.collection.get_doc(docid)


