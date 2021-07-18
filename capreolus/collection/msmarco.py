import os
import gzip
import shutil
import tarfile
from time import time

from capreolus import constants
from capreolus.utils.common import download_file
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import document_to_trectxt
from . import Collection

logger = get_logger(__name__)
PACKAGE_PATH = constants["PACKAGE_PATH"]


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
            elif not os.path.isdir(tmp_dir):  # and tmp_dir != list(expected_fns)[0]:
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
    # is_large_collection = True

    def download_raw(self):
        url = "https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz"
        tmp_dir = self.get_cache_path() / "tmp"
        expected_fns = {
            "collection.tsv",
            "qrels.dev.small.tsv",
            "qrels.train.tsv",
            "queries.train.tsv",
            "queries.dev.small.tsv",
            "queries.dev.tsv",
            "queries.eval.small.tsv",
            "queries.eval.tsv",
        }
        gz_dir = self.download_and_extract(url, tmp_dir, expected_fns=expected_fns)
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
class MSMARCO_DOC_V2(Collection):
    module_name = "msdoc_v2"
    collection_type = "MsMarcoDocV2Collection"
    generator_type = "DefaultLuceneDocumentGenerator"
    data_dir = PACKAGE_PATH / "data" / "msdoc_v2"
    _path = data_dir / "msmarco_v2_doc"
    # is_large_collection = True


@Collection.register
class MSMARCO_DOC_V2_Presegmented(Collection):
    """ This colletion share exactly the same qrels, topic and folds with MS MARCO v2 """
    module_name = "msdoc_v2_preseg"
    collection_type = "MsMarcoDocV2Collection"
    generator_type = "DefaultLuceneDocumentGenerator"
    data_dir = PACKAGE_PATH / "data" / "msdoc_v2"
    _path = data_dir / "msmarco_v2_doc_segmented"


@Collection.register
class MSMARCO_PSG_V2(Collection):
    module_name = "mspsg_v2"
    collection_type = "MsMarcoPassageV2Collection"
    generator_type = "DefaultLuceneDocumentGenerator"
    data_dir = PACKAGE_PATH / "data" / "mspass_v2"
    _path = data_dir / "msmarco_v2_passage"
    # is_large_collection = True
