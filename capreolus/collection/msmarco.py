import gzip
import shutil
from time import time

from capreolus.utils.common import download_file
from capreolus.utils.loginit import get_logger
from . import Collection

logger = get_logger(__name__)


@Collection.register
class MSMarco(Collection):
    module_name = "msmarco"
    config_keys_not_in_path = ["path"]
    collection_type = "TrecCollection"
    generator_type = "DefaultLuceneDocumentGenerator"
    # config_spec = [ConfigOption("path", "/GW/NeuralIR/nobackup/msmarco/trec_format", "path to corpus")]

    def download_if_missing(self):
        coll_dir = self.get_cache_path() / "documents"
        coll_fn = coll_dir / "msmarco.collection.txt"
        if coll_fn.exists():
            return coll_dir

        coll_url = "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.trec.gz"
        tmp_coll_fn = self.get_cache_path() / "tmp" / "msmarco-docs.trec.gz"
        if not tmp_coll_fn.exists():
            tmp_coll_fn.parent.mkdir(exist_ok=True, parents=True)
            download_file(coll_url, tmp_coll_fn)

        coll_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Extracting {tmp_coll_fn}...")
        t = time()
        with gzip.open(tmp_coll_fn, "rb") as fin, open(coll_fn, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        duration = int(time() - t)
        min, sec = duration // 60, duration % 60
        logger.info(f"{tmp_coll_fn} extracted after {duration} seconds (00:{min}:{sec})")
        return coll_dir
