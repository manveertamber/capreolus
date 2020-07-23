import gzip
import shutil

from capreolus.utils.common import download_file
from capreolus.utils.loginit import get_logger
from . import Collection


@Collection.register
class MSMarco(Collection):
    module_name = "msmarco"
    config_keys_not_in_path = ["path"]
    collection_type = "TrecCollection"
    generator_type = "DefaultLuceneDocumentGenerator"
    # config_spec = [ConfigOption("path", "/GW/NeuralIR/nobackup/msmarco/trec_format", "path to corpus")]

    def download_if_missing(self):
        coll_fn = self.get_cache_path() / "documents"
        if coll_fn.exists():
            return coll_fn

        coll_url = "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.trec.gz"
        tmp_coll_fn = self.get_cache_path() / "tmp" / "msmarco-docs.trec.gz"
        if not tmp_coll_fn.exists():
            tmp_coll_fn.parent.mkdir(exist_ok=True, parents=True)
            download_file(coll_url, tmp_coll_fn)

        coll_fn.parent.mkdir(exist_ok=True, parents=True)
        with gzip.open(tmp_coll_fn, "r") as fin, open(coll_fn, "w", encoding="utf-8") as fout:
            shutil.copyfileobj(fin, fout)

        return coll_fn
