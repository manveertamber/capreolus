import os
import shutil
import tarfile
from time import time

import cbor
from capreolus import ConfigOption, constants
from capreolus.utils.common import download_file
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import anserini_index_to_trec_docs
from capreolus.collection.helpers.trec_car_classes import Page

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

    is_large_collection = True

    def build(self):
        self.hierarchize_docs()

    @staticmethod
    def get_page_subpath(pid):
        code = int(pid, 16)
        fir, sec = str(code % (10 ** 2)), str(code % (10 ** 4))[:2]
        return f"{fir}/{sec}.cbor"

    @staticmethod
    def itercbor(cbor_fn):
        with open(cbor_fn) as f:
            while True:
                try:
                    yield cbor.load(f)
                except EOFError:
                    break

    def hierarchize_docs(self):
        # Iterate over the entire documents and separate it into 100/100 directorys (like GOV2)
        # according to its 16-digits int expression
        # then we only need to search for its file when fetching doc at training time
        # Note that we do not record its byte position since it's not sure
        # how to get the number of bytes from the "cbor.load()"-ed object:
        # the value returned by sys.getsizeof() contain the overhead of meta-information
        # while cbor cannot be read using "readline()" etc. so we do not know dircectly
        # how much bytes to read for each passage (unless we re-implement the content of cbor.load()
        path, _, _ = self.get_path_and_types()
        splitpath = path / "split"
        for subdir in os.listdir(splitpath):
            shutil.rmtree(path / subdir)
            logger.info(f"Existing folder {path / subdir} has been removed before preparing new collections")

        t = time()
        for i, (tag, pid, psgs) in enumerate(self.itercbor(path / "paragraphcorpus" / "paragraphcorpus.cbor")):
            if i and i % 30000 == 0:
                len_info = [len(vv) for k, v in hierachy.items() for kk, vv in v.items()]
                logger.info(
                    f"{i} documents prepared in {time() - t} sec, \n\t"
                    f"avg #doc per file : {np.mean(len_info)} "
                    f"max #doc per file: {max(len_info)} "
                    f"min #doc per file: {min(len_info)}")

            fn = splitpath / self.get_page_subpath(pid)
            # mode = "ab" if fn.exist() else "wb"
            fn.parent.mkdir(exist_ok=True, parents=True)
            cbor.dump([tag, pid, psgs], open(fn, "ab"))  # mode

    def get_doc(self, doc_id):
        path, _, _ = self.get_path_and_types()
        fn = path / "split" / self.get_page_file(doc_id)
        for tag, pid, psgs in self.itercbor(fn):
            if pid != doc_id:
                continue
            return Page.from_cbor([tag, pid, psgs])

    def download_if_missing(self):
        url = "http://trec-car.cs.unh.edu/datareleases/v1.5/paragraphcorpus-v1.5.tar.xz"
        tmpfile = self.get_cache_path() / "tmp" / url.split("/")[-1]
        target_dir = self.get_cache_path() / "documents"
        tmpfile.parent.mkdir(exist_ok=True, parents=True)
        target_dir.mkdir(exist_ok=True, parents=True)

        if target_dir.exists() and os.listdir(target_dir):
            return target_dir

        if not tmpfile.exists():
            download_file(url, tmpfile)
        with tarfile.open(tmpfile) as tarobj:
            tarobj.extractall(target_dir)

        return target_dir
