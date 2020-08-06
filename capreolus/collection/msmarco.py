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


def _download_and_extract(url, tmp_dir):
    tmp_dir.mkdir(exist_ok=True, parents=True)
    gz_name = url.split("/")[-1]
    output_gz = tmp_dir / gz_name
    if not output_gz.exists():
        output_gz.parent.mkdir(exist_ok=True, parents=True)
        download_file(url, output_gz)

    logger.info(f"Extracting {output_gz}...")
    t = time()
    if str(output_gz).endswith("tar.gz"):
        tmp_dir = tmp_dir / gz_name.replace(".tar.gz", "")
        expected_fns = {"collection.tsv", "qrels.dev.small.tsv", "qrels.train.tsv", "queries.train.tsv",
            "queries.dev.small.tsv", "queries.dev.tsv", "queries.eval.small.tsv", "queries.eval.tsv"}
        if not (tmp_dir.exists() and set(os.listdir(tmp_dir)) == expected_fns):
            with tarfile.open(output_gz, "r:gz") as f:
                f.extractall(path=tmp_dir)
    else:
        outp_fn = tmp_dir / gz_name.replace(".gz", "")
        if not outp_fn.exists():
            with gzip.open(output_gz, "rb") as fin, open(outp_fn, "wb") as fout:
                shutil.copyfileobj(fin, fout)

    duration = int(time() - t)
    min, sec = duration // 60, duration % 60
    logger.info(f"{output_gz} extracted after {duration} seconds (00:{min}:{sec})")
    return tmp_dir


@Collection.register
class MSMarcoPsg(Collection):
    module_name = "msmarcopsg"
    collection_type = "TrecCollection"
    generator_type = "DefaultLuceneDocumentGenerator"

    def download_raw(self):
        # url = "https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz"
        url = "https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz"
        tmp_dir = self.get_cache_path() / "tmp"
        gz_dir =_download_and_extract(url, tmp_dir)
        return gz_dir

    def download_if_missing(self):
        coll_dir = self.get_cache_path() / "documents"
        coll_fn = coll_dir / "msmarco.psg.collection.txt"
        if coll_fn.exists():
            return coll_dir

        # convert to trec file
        coll_tsv_fn = self.download_raw() / "collection.tsv"
        coll_fn.parent.mkdir(exist_ok=True, parents=True)
        with open(coll_tsv_fn, "r") as fin, open(coll_fn, "w", encoding="utf-8") as fout:
            for line in fin:
                docid, doc = line.strip().split("\t")
                fout.write(document_to_trectxt(docid, doc))

        return coll_dir


@Collection.register
class MSMarcoDoc(Collection):
    module_name = "msmarcodoc"
    config_keys_not_in_path = ["path"]
    collection_type = "TrecCollection"
    generator_type = "DefaultLuceneDocumentGenerator"

    def download_raw(self):
        url = "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.trec.gz"
        tmp_dir = self.get_cache_path() / "tmp"
        gz_dir =_download_and_extract(url, tmp_dir)
        return gz_dir

    def download_if_missing(self):
        coll_dir = self.get_cache_path() / "documents"
        coll_fn = coll_dir / "msmarco.doc.collection.txt"
        if coll_fn.exists():
            return coll_dir

        coll_tmp_fn = self.download_raw() / "msmarco-docs.trec"
        shutil.move(coll_tmp_fn, coll_fn)
        return coll_dir
