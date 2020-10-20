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

            if os.path.isdir(tmp_dir): # and set(os.listdir(tmp_dir)) != expected_fns:
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

    def download_raw(self):
        url = "https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz"
        tmp_dir = self.get_cache_path() / "tmp"
        expected_fns = {"collection.tsv", "qrels.dev.small.tsv", "qrels.train.tsv", "queries.train.tsv", "queries.dev.small.tsv", "queries.dev.tsv", "queries.eval.small.tsv", "queries.eval.tsv"}
        gz_dir = self.download_and_extract(url, tmp_dir, expected_fns=expected_fns)
        return gz_dir

    def download_if_missing(self):
        self.tsv_folder = self.get_cache_path() / "subfolders"
        coll_fn = self.tsv_folder
        if coll_fn.exists():
            return coll_fn

        # convert to trec file
        coll_tsv_fn = self.download_raw() / "collection.tsv"
        coll_fn.parent.mkdir(exist_ok=True, parents=True)
        '''
        with open(coll_tsv_fn, "r") as fin, open(coll_fn, "w", encoding="utf-8") as fout:
            for line in fin:
                docid, doc = line.strip().split("\t")
                fout.write(document_to_trectxt(docid, doc))
        '''

        # transform the msmarco into gov2 format
        root, folder, subfolder = "000", "000", "000"
        fn = self.tsv_folder / root / folder / subfolder
        fn.mkdir(exist_ok=True, parents=True)
        outp_file = open(fn, "w")
        with open(coll_tsv_fn) as f:
            for line in f:
                docid, doc = line.split("\t")
                cur_root, cur_folder, cur_subfolder = \
                    self.idx2foldername(int(docid) % (1000**3)), \
                    self.idx2foldername(int(docid) % (1000**2)), \
                    self.idx2foldername(int(docid) % 1000)

                if (cur_root != root) or (folder != cur_folder) or (subfolder != subfolder):
                    print(cur_root, cur_folder, cur_subfolder)
                    outp_file.close()
                    root, folder, subfolder = cur_root, cur_folder, cur_subfolder
                    fn = self.tsv_folder / root / folder / subfolder
                    fn.mkdir(exist_ok=True, parents=True)
                    outp_file = open(fn, "w")

                outp_file.write(f"{docid}\t{doc}")

        return coll_fn

    def get_doc(self, docid):
        if not hasattr(self, "tsv_folder") or not os.path.exists(self.tsv_folder):
            self.download_if_missing()
            # self.tsv_folder = self.get_cache_path() / "subfolders"

        root, folder, subfolder = \
            self.idx2foldername(int(docid) % (1000 ** 3)), \
            self.idx2foldername(int(docid) % (1000 ** 2)), \
            self.idx2foldername(int(docid) % 1000)
        path = self.tsv_folder / root / folder / subfolder
        return self.find_doc_in_single_file(path, docid)

    @staticmethod
    def idx2foldername(idx):
        idx_str = str(idx)
        return "0" * (3 - len(idx_str)) + idx_str

    @staticmethod
    def find_doc_in_single_file(filename, docid):
        with open(filename) as f:
            for line in f:
                if line.startswith(docid):
                    i, doc = line.strip().split("\t")
                    assert i == docid
                    return doc
        return ""

