import os
import stat
import gzip
import json
import shutil
import tarfile
from time import time
from collections import defaultdict

from tqdm import tqdm
from capreolus import Dependency, constants
from capreolus.utils.common import download_file
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import document_to_trectxt
from . import Collection, IRDCollection

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
class MSMARCO_DOC_V2_Presegmented(Collection):
    """This colletion share exactly the same qrels, topic and folds with MS MARCO v2"""

    module_name = "msdoc_v2_preseg"
    collection_type = "MsMarcoDocV2Collection"
    generator_type = "DefaultLuceneDocumentGenerator"
    data_dir = PACKAGE_PATH / "data" / "msdoc_v2"
    _path = data_dir / "msmarco_v2_doc_segmented"
    url = "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco_v2_doc_segmented.tar"

    def build(self):
        self.download_if_missing()

        collection_path = self.get_path_and_types()[0]
        self.id2pos_map
        self.cache = {}  # docid to doc
        self.name2filehandle = {psg_fn: open(collection_path / psg_fn, "rt", encoding="utf-8") for psg_fn in self.id2pos_map}

    def __del__(self):
        if hasattr(self, "name2filehandle"):
            for file_handler in self.name2filehandle.values():
                file_handler.close()

    def download_if_missing(self):
        if self._path.exists() and len(os.listdir(self._path)) == 60:
            return

        tmp_path = self.get_cache_path() / "tmp"
        cache_path = self.get_cache_path() / "document"
        tmp_path.mkdir(parents=True, exist_ok=True)
        cache_path.mkdir(parents=True, exist_ok=True)
        extract_dir = cache_path / "msmarco_v2_doc_segmented"

        tarball_name = self.url.split("/")[-1]
        msmarco_v2_doc_segmented_tarball = tmp_path / tarball_name
        # todo: this fails a lot
        download_file(self.url, msmarco_v2_doc_segmented_tarball, expected_hash="f18c3a75eb3426efeb6040dca3e885dc", hash_type="md5")

        def unzip_gz(gz_file):
            import gzip
            import shutil
            ungz_file = gz_file.as_posix()[:-3]
            with gzip.open(gz_file, "rb") as f_in:
                with open(ungz_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(f_in)

        if not (extract_dir.exists() and len(os.listdir(extract_dir)) == 60):
            with tarfile.open(msmarco_v2_doc_segmented_tarball) as f:
                f.extractall(cache_path)

        extract_dir.chmod(stat.S_IRWXU | stat.S_IRWXG)
        for gz_file in extract_dir.iterdir():
            if gz_file.suffix == ".gz":
                unzip_gz(gz_file)

        self._path.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(extract_dir, self._path)
        logger.info(f"MS MARCO Pre-segmented collection prepared under {self._path}.")


    @staticmethod
    def build_id2pos_map(json_gz_file):
        assert json_gz_file.suffix == ".jsonl"
        id2pos = defaultdict(dict)
        idx, interval = 0, 1_000_000
        with open(json_gz_file) as f:
            while True:
                if idx % interval == 0 and idx > 0:
                    logger.info(f"[%.2f M] lines processed in file {os.path.basename(json_gz_file)}" % (idx / 1_000_000))

                pos = f.tell()
                line = f.readline()
                if line == "":
                    break

                docid = json.loads(line)["docid"]
                assert "#" in docid
                root_docid, suffix = docid.split("#")
                id2pos[root_docid][suffix] = pos
                idx += 1
        return id2pos

    @property
    def id2pos_map(self):
        if hasattr(self, "_id2pos_map"):
            return self._id2pos_map

        cache_path = self.get_cache_path()
        id2pos_map_path = cache_path / "id2pos_map.json"

        if id2pos_map_path.exists():
            self._id2pos_map = json.load(open(id2pos_map_path))
            return self._id2pos_map

        collection_path = self.get_path_and_types()[0]
        self._id2pos_map = {
            json_gz_file.name: self.build_id2pos_map(json_gz_file)
            for json_gz_file in collection_path.iterdir()
            if json_gz_file.suffix == ".jsonl"
        }
        json.dump(self._id2pos_map, open(id2pos_map_path, "w"))
        return self._id2pos_map

    def get_doc(self, docid):
        if docid in self.cache:
            return self.cache[docid]

        for psg_fn in self.id2pos_map:
            root_docid, suffix = docid.split("#")
            if root_docid in self.id2pos_map[psg_fn]:

                position = self.id2pos_map[psg_fn][root_docid][suffix]
                in_fh = self.name2filehandle[psg_fn]
                in_fh.seek(position)

                doc = json.loads(in_fh.readline())
                doc["body"] = " ".join(doc["body"].split())

                assert doc["docid"] == docid
                self.cache[docid] = doc
                return doc

    def get_passages(self, docid):
        if "#" in docid:
            return [self.get_doc(docid)]

        assert "#" not in docid
        collection_path = self.get_path_and_types()[0]
        passages = []

        for psg_fn in self.id2pos_map:
            if docid in self.id2pos_map[psg_fn]:

                position = self.id2pos_map[psg_fn][docid]["0"]
                in_fh = self.name2filehandle[psg_fn]
                in_fh.seek(position)

                for suffix in self.id2pos_map[psg_fn][docid]:
                    doc = json.loads(in_fh.readline())
                    doc["body"] = " ".join(doc["body"].split())
                    assert doc["docid"] == f"{docid}#{suffix}", f"Expect {docid}#{suffix} but got {doc['docid']}"
                    passages.append(doc)

        return passages


@Collection.register
class MSMARCO_DOC_V2(IRDCollection):
    """ MS MARCO Document v2: https://microsoft.github.io/msmarco/TREC-Deep-Learning.html#document-ranking-dataset """

    module_name = "msdoc_v2"
    ird_dataset_name = "msmarco-document-v2"
    collection_type = "JsonCollection"

    dependencies = [
        Dependency(key="collection", module="collection", name="msdoc_v2_preseg"),  # need msdoc_v2_preseg to prepare passage
    ]

    def doc_as_json(self, doc):
        # content = " ".join((doc.headline, doc.body))
        # todo: able to control the field
        return json.dumps({"id": doc.doc_id, "contents": doc.body})

    def get_doc(self, docid):
        return self.docs_store.get(docid).text

    def get_passages(self, docid):
        return self.collection.get_passages(docid)


@Collection.register
class MSMARCO_PSG_V2(IRDCollection):
    """ MS MARCO Passage v2: https://microsoft.github.io/msmarco/TREC-Deep-Learning.html#passage-ranking-dataset """

    module_name = "mspsg_v2"
    ird_dataset_name = "msmarco-passage-v2"
    collection_type = "JsonCollection"

    def doc_as_json(self, doc):
        return json.dumps({"id": doc.doc_id, "contents": doc.text})

    def get_doc(self, docid):
        return self.docs_store.get(docid).text

    def get_passages(self, docid):
        return [self.get_doc(docid)]
