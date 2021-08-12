import os
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

    dependencies = [
        # need msdoc_v2_preseg to prepare passage 
        Dependency(key="collection", module="collection", name="msdoc_v2_preseg"),
    ]

    def get_doc(self, docid):
        collection_path = self.get_path_and_types()[0]

        (string1, string2, bundlenum, position) = docid.split("_")
        assert string1 == "msmarco" and string2 == "doc"
        with gzip.open(collection_path / f"msmarco_doc_{bundlenum}.gz", "rt", encoding="utf8") as in_fh:
            in_fh.seek(int(position))
            json_string = in_fh.readline()
            doc = json.loads(json_string)
            assert doc["docid"] == docid
        return doc
    
    def get_passages(self, docid):
        return self.collection.get_passages(docid)


@Collection.register
class MSMARCO_DOC_V2_Presegmented(Collection):
    """This colletion share exactly the same qrels, topic and folds with MS MARCO v2"""

    module_name = "msdoc_v2_preseg"
    collection_type = "MsMarcoDocV2Collection"
    generator_type = "DefaultLuceneDocumentGenerator"
    data_dir = PACKAGE_PATH / "data" / "msdoc_v2"
    _path = data_dir / "msmarco_v2_doc_segmented"

    def build(self):
        collection_path = self.get_path_and_types()[0]

        self.id2pos_map
        self.cache = {}  # docid to doc
        self.name2filehandle = {psg_fn: open(collection_path / psg_fn, "rt", encoding="utf-8") for psg_fn in self.id2pos_map}

    def __del__(self):
        if hasattr(self, "name2filehandle"):
            for file_handler in self.name2filehandle.values():
                file_handler.close()

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
class MSMARCO_PSG_V2(Collection):
    module_name = "mspsg_v2"
    collection_type = "MsMarcoPassageV2Collection"
    generator_type = "DefaultLuceneDocumentGenerator"
    data_dir = PACKAGE_PATH / "data" / "mspass_v2"
    _path = data_dir / "msmarco_v2_passage"
    # is_large_collection = True

    def build(self):
        collection_path = self.get_path_and_types()[0]
        self.cache = {}  # docid to doc
        self.name2filehandle = {
            psg_fn.name: open(psg_fn, "rt", encoding="utf-8") for psg_fn in collection_path.iterdir()
        }  # the files under input folder must be uncompressed to support fast f.seek

    def __del__(self):
        for file_handler in self.name2filehandle.values():
            file_handler.close()

    @staticmethod
    def read_all_before_pos(in_fh, pos, cache):
        cur_passage = None
        in_fh.seek(0)
        while True:
            cur_pos = in_fh.tell()
            if cur_pos > pos:
                break

            line = json.loads(in_fh.readline())
            pid, passage = line["pid"], " ".join(line["passage"].split())

            if pid not in cache:
                cache[pid] = passage

            if cur_pos == pos:
                cur_passage = passage

        return cur_passage

    def get_doc(self, docid):
        if docid in self.cache:
            return self.cache[docid]

        collection_path = self.get_path_and_types()[0]
        (string1, string2, bundlenum, position) = docid.split("_")
        position = int(position)
        assert string1 == "msmarco" and string2 == "passage"

        psg_fn = f"msmarco_passage_{bundlenum}"
        in_fh = self.name2filehandle[psg_fn]
        in_fh.seek(position)
        doc = json.loads(in_fh.readline())
        assert doc["pid"] == docid

        passage = " ".join(doc["passage"].split())
        self.cache[docid] = passage

        return passage
