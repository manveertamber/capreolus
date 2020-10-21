import os
import gzip
import shutil
import tarfile
from time import time
import re

from capreolus.utils.common import download_file
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import document_to_trectxt
from . import Collection

logger = get_logger(__name__)


class MSMarcoMixin:
    @staticmethod
    def download_and_extract(url, tmp_dir, expected_fns=None):
        tmp_dir.mkdir(exist_ok=True, parents=True)
        gz_name = url.split("/")[-1]
        output_gz = tmp_dir / gz_name
        if not output_gz.exists():
            logger.info(f"Downloading from {url}...")
            download_file(url, output_gz)

        extract_dir = None
        t = time()
        if str(output_gz).endswith("tar.gz"):
            tmp_dir = tmp_dir / gz_name.replace(".tar.gz", "")
            logger.info(f"tmp_dir: {tmp_dir}")
            if not tmp_dir.exists():
                logger.info(f"{tmp_dir} file does not exist, extracting from {output_gz}...")
                with tarfile.open(output_gz, "r:gz") as f:
                    f.extractall(path=tmp_dir)

            if os.path.isdir(tmp_dir):  # and set(os.listdir(tmp_dir)) != expected_fns:
                extract_dir = tmp_dir
            elif (not os.path.isdir(tmp_dir)): # and tmp_dir != list(expected_fns)[0]:
                extract_dir = tmp_dir.parent

        else:
            outp_fn = tmp_dir / gz_name.replace(".gz", "")
            if not outp_fn.exists():
                logger.info(f"{tmp_dir} file does not exist, extracting from {output_gz}...")
                with gzip.open(output_gz, "rb") as fin, open(outp_fn, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
            extract_dir = tmp_dir

        duration = int(time() - t)
        min, sec = duration // 60, duration % 60
        logger.info(f"{output_gz} extracted after {duration} seconds (00:{min}:{sec})")
        return extract_dir


@Collection.register
class MSMarcoPsg(Collection, MSMarcoMixin):
    module_name = "msmarcopsg"
    collection_type = "TrecCollection"
    generator_type = "DefaultLuceneDocumentGenerator"
    is_large_collection = True
    regex = re.compile('[^a-zA-Z0-9]')

    def download_raw(self):
        url = "https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz"
        tmp_dir = self.get_cache_path() / "tmp"
        expected_fns = {"collection.tsv", "qrels.dev.small.tsv", "qrels.train.tsv", "queries.train.tsv", "queries.dev.small.tsv", "queries.dev.tsv", "queries.eval.small.tsv", "queries.eval.tsv"}
        gz_dir = self.download_and_extract(url, tmp_dir, expected_fns=expected_fns)
        return gz_dir

    def download_if_missing(self):
        self.tsv_folder = self.get_cache_path() / "subfolders"
        coll_fn = self.tsv_folder
        if (coll_fn / "done").exists():
            return coll_fn

        # convert to trec file
        coll_tsv_fn = self.download_raw() / "collection.tsv"
        coll_fn.parent.mkdir(exist_ok=True, parents=True)

        # transform the msmarco into gov2 format
        fn = self.get_file_path(docid=0)
        fn.parent.mkdir(exist_ok=True, parents=True)
        outp_file = open(fn, "w")
        with open(coll_tsv_fn) as f:
            for line in f:
                docid, doc = line.split("\t")
                cur_fn = self.get_file_path(docid=docid)
                if cur_fn != fn:
                    outp_file.close()
                    fn = cur_fn
                    fn.parent.mkdir(exist_ok=True, parents=True)
                    outp_file = open(fn, "w")

                outp_file.write(f"{docid}\t{doc}")

        with open(coll_fn / "done", "w") as f:
            f.write("done")

        return coll_fn

    def get_doc(self, docid):
        if not hasattr(self, "tsv_folder") or not os.path.exists(self.tsv_folder):
            self.download_if_missing()

        path = self.get_file_path(docid)
        doc = self.find_doc_in_single_file(path, docid)
        if not doc:
            logger.warning(f"{docid} cannot be found from collection")
        doc = " ".join(self.regex.sub(" ", doc).split())
        return doc

    @staticmethod
    def find_doc_in_single_file(filename, docid):
        with open(filename) as f:
            for line in f:
                if line.startswith(docid):
                    i, doc = line.strip().split("\t")
                    assert i == docid
                    return doc
        return ""

    def get_file_path(self, docid):
        # assume max digits is 9
        docid = str(docid)
        docid = "0" * (9 - len(docid)) + docid
        path = self.tsv_folder / docid[:3] / docid[3:6]
        return path
