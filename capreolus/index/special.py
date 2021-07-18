import math
import os
import subprocess

from capreolus import Dependency, ConfigOption, constants, get_logger
from capreolus.utils.common import Anserini

from . import Index

logger = get_logger(__name__)  # pylint: disable=invalid-name
MAX_THREADS = constants["MAX_THREADS"]


@Index.register
class MsV2Index(Index):
    """This index read documents from msmarco doc v2 colleciton or presegmented doc v2 collection"""

    module_name = "msdoc_v2"
    config_spec = [
        ConfigOption("fields", ["url", "title", "headings", "body"], "which fields to include", value_type="strlist"),
    ]

    @staticmethod
    def build_id2pos_map_single(json_gz_file):
        assert json_gz_file.suffix == ".gz" and json_gz_file.suffix.suffix == ".json"
        self.id2pos = {}
        with gzip.open(json_gz_file) as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if line == "":
                    break

                docid = json.loads(line)["docid"]
                assert "#" in docid
                id2pos[docid] = pos

    @property
    def id2pos_map(self):
        if hasattr(self, "_id2pos_map"):
            return self._id2pos_map

        cache_path = self.get_cache_path()
        id2pos_map_path = cache_path / "id2pos_map.json"

        if id2pos_map_path.exists():
            return json.load(open(id2pos_map_path))

        collection_path = self.collection.get_path_and_types()[0]
        self._id2pos_map = {
            json_gz_file: self.build_id2pos_map_single(json_gz_file) for json_gz_file in collection_path.iterdir()
        }
        json.dump(self._id2pos_map, open(id2pos_map_path, "w"))
        return self._id2pos_map

    def _create_index(self):
        collection = self.collection.module_name
        supported_collections = ["msdoc_v2", "msdoc_v2_preseg"]
        if collection not in supported_collections:
            raise ValueError(f"Not supported collection module: {collection}, should be one of {supported_collections}.")

        if collection == "msdoc_v2":
            return

        self.id2pos_map

    def get_docs(self, doc_ids):
        return [self.get_doc(doc_id) for doc_id in doc_ids]

    def get_doc(self, docid):
        collection = self.collection.module_name
        if "preseg" in collection:
            return self.get_doc_from_msdoc_presegmented(docid)
        else:
            return self.get_doc_from_msdoc(docid)

    def get_doc_from_msdoc(self, docid):
        collection_path = self.collection.get_path_and_types()[0]

        (string1, string2, bundlenum, position) = docid.split("_")
        assert string1 == "msmarco" and string2 == "doc"
        with gzip.open(collection_path / f"msmarco_doc_{bundlenum}.gz", "rt", encoding="utf8") as in_fh:
            in_fh.seek(int(position))
            json_string = in_fh.readline()
            doc = json.loads(json_string)
            assert doc["docid"] == docid
            return " ".join([doc.get(field, "") for field in self.config["fields"]])

    def get_doc_from_msdoc_presegmented(self, docid):
        collection_path = self.collection.get_path_and_types()[0]

        for psg_fn in self.id2pos_map:
            if docid in self.id2pos_map[psg_fn]:
                position = self.id2pos_map[psg_fn][docid]
                in_fh = gzip.open(collection_path / psg_fn, "rt", encoding="utf-8")
                in_fh.seek(position)
                doc = json.loads(in_fh.readline())
                assert doc["docid"] == docid
                if "segment" in doc and "body" not in doc:
                    # replace the body in field with segment
                    fields = [field if field != "body" else "segment" for field in self.config["fields"]]
                return " ".join([doc.get(field, "") for field in fields])

    def get_df(self, term):
        return None

    def get_idf(self, term):
        return None


@Index.register
class MsPsgV2Index(Index):
    """This index read documents from msmarco doc v2 colleciton or presegmented doc v2 collection"""

    module_name = "mspsg_v2"
    dependencies = [
        Dependency(key="collection", module="collection", name="mspsg_v2"),
    ]

    def _create_index(self):
        collection = self.collection.module_name
        if collection != "msdoc_v2_preseg":
            raise ValueError(f"Not supported collection module: {collection}, should msdoc_v2_preseg.")

    def get_docs(self, doc_ids):
        return [self.get_doc(doc_id) for doc_id in doc_ids]

    def get_doc(self, docid):
        collection_path = self.collection.get_path_and_types()[0]

        (string1, string2, bundlenum, position) = sys.argv[i].split("_")
        assert string1 == "msmarco" and string2 == "passage"
        with gzip.open(collection_path / f"msmarco_passage_{bundlenum}.gz", "rt", encoding="utf8") as in_fh:
            in_fh.seek(int(position))
            json_string = in_fh.readline()
            doc = json.loads(json_string)
            assert doc["pid"] == pid
            return doc["passage"]

    def get_df(self, term):
        return None

    def get_idf(self, term):
        return None
