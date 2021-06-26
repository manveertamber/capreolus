import os
import gzip
import shutil
import tarfile
from time import time

from capreolus.utils.common import download_file
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import document_to_trectxt
from . import Collection

logger = get_logger(__name__)



@Collection.register
class MSMarcoPsg(Collection):
    module_name = "msdoc_v2"
    collection_type = "TrecCollection"
    generator_type = "DefaultLuceneDocumentGenerator"
    # is_large_collection = True

