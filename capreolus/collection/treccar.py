import os
import tarfile

from capreolus import ConfigOption, constants
from capreolus.utils.common import download_file
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import anserini_index_to_trec_docs

from . import Collection

logger = get_logger(__name__)
PACKAGE_PATH = constants["PACKAGE_PATH"]


@Collection.register
class TRECCAR(Collection):
    """ TREC Car """
    module_name = "treccar"
    collection_type = "CarCollection"
    generator_type = "DefaultLuceneDocumentGenerator"
    config_keys_not_in_path = ["path"]
    config_spec = [
        ConfigOption("path", "", "path to corpus")]

    def download_if_missing(self):
        url = "http://trec-car.cs.unh.edu/datareleases/v1.5/paragraphcorpus-v1.5.tar.xz"
        tmpfile = self.get_cache_path() / "tmp" / url.split("/")[-1]
        target_dir = self.get_cache_path() / "documents"

        print("tmp collection: ", tmpfile)
        print("target collection: ", target_dir)
        exit(0)

        if target_dir.exists() and os.listdir(target_dir):
            return target_dir

        if not tmpfile.exists():
            download_file(url, tmpfile)
        with tarfile.open(tmpfile) as tarobj:
            tarobj.extract(target_dir)

        return target_dir
