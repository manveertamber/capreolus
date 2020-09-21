import os
import sys
import gzip
import json
import math
import random
from time import time
from multiprocessing import Manager, Pool
from pathlib import Path
import numpy as np

from lxml.html.clean import Cleaner  # remove javascript
from lxml.html import fromstring

from capreolus import ConfigOption, Dependency, constants, parse_config_string
from capreolus.collection import Collection

from capreolus.utils.loginit import get_logger


DOC, TERMINATING_DOC = "<DOC>", "</DOC>"
DOCNO, TERMINATING_DOCNO = "<DOCNO>", "</DOCNO>"
DOCHDR, TERMINATING_DOCHDR = "<DOCHDR>", "</DOCHDR>"

logger = get_logger(__name__)


def clean_documents(docs):
    docs = docs.replace(f"{DOC}\n", "").replace(f"{TERMINATING_DOC}\n", "")
    return [" ".join(d.split()).strip() for d in docs.split("\n")]


def get_next_record_pos(f):
    start_pos = f.tell()
    found = False
    line, doc_lines = f.readline(), []
    line = line.decode("utf-8")
    while line != "":
        if line.startswith(DOC):
            found = True

        if found:
            doc_lines.append(line)

        if line.startswith(TERMINATING_DOC) and len(doc_lines) != 0:
            doc_lines = clean_documents("".join(doc_lines))
            assert doc_lines[0].startswith(DOCNO)
            id = parse_record(doc_lines)
            return id, (start_pos, f.tell() - start_pos)

        line = f.readline().decode("utf-8", errors="ignore")

    return "END", (-1, -1)


def parse_record(doc_lines, return_doc=False):
    if isinstance(doc_lines, list):
        doc = " ".join(doc_lines)
    else:
        doc = doc_lines

    i = doc.index(DOCNO)
    if i == -1:
        raise ValueError("cannot find start tag " + DOCNO)
    if i != 0:
        raise ValueError("should start with " + DOCNO)

    j = doc.index(TERMINATING_DOCNO)
    if j == -1:
        raise ValueError("cannot find end tag " + TERMINATING_DOCNO)

    id = doc[i+len(DOCNO):j].replace(DOCNO, "").replace(TERMINATING_DOCNO, "")

    if not return_doc:
        return id

    i = doc.index(DOCHDR)
    if i == -1:
        raise ValueError("cannot find header tag " + DOCHDR)

    j = doc.index(TERMINATING_DOCHDR)
    if j == -1:
        raise ValueError("cannot find end tag " + TERMINATING_DOCHDR)
    if j < i:
        raise ValueError(TERMINATING_DOCHDR + " comes before " + DOCHDR)

    # doc = doc[i+len(DOCHDR):j].replace(DOCHDR, "").replace(TERMINATING_DOCHDR, "").strip()
    doc = doc[j+len(TERMINATING_DOCHDR):].replace(TERMINATING_DOCHDR, "").replace(TERMINATING_DOC, "").strip()
    # print(len(doc), doc[-100:])
    return id, doc


def spawn_child_process_to_read_docs(data):
    path, shared_id2pos = data["path"], data["shared_id2pos"]

    start = time()
    local_id2pos = {}
    for gz_fn in os.listdir(path):
        with gzip.open(f"{path}/{gz_fn}") as f:
            id, pos = get_next_record_pos(f)
            while id != "END":
                local_id2pos[id] = pos
                id, pos = get_next_record_pos(f)

    shared_id2pos.update(local_id2pos)
    logger.info("PID: {0}, Done getting documents from disk: {1} for path: {2}".format(os.getpid(), time() - start, path))


@Collection.register
class GOV2Collection(Collection):
    path = "/GW/NeuralIR/nobackup/GOV2/GOV2_data"  # under this file should be a list of

    collection_type = "TrecwebCollection"
    generator_type = "DefaultLuceneDocumentGenerator"
    module_name = "gov2collection"

    BUFFER_SIZE = 500

    @property
    def id2pos(self):
        if not hasattr(self, "_id2pos"):
            self.prepare_id2pos()
        return self._id2pos

    def prepare_id2pos(self):
        self.file_buffer = {}

        cache_dir = self.get_cache_path()
        cache_dir.mkdir(exist_ok=True, parents=True)
        id2pos_fn = cache_dir / "id2pos.json"
        if id2pos_fn.exists():
            logger.info(f"collection docid2pos found under {id2pos_fn}")
            self._id2pos = json.load(open(id2pos_fn))
            return

        rootdir = self.path
        manager = Manager()
        shared_id2pos = manager.dict()
        all_dirs = [f"{rootdir}/{subdir}" for subdir in os.listdir(rootdir) if os.path.isdir(rootdir + "/" + subdir)]
        args_list = [{"path": folder, "shared_id2pos": shared_id2pos} for folder in sorted(all_dirs)]
        logger.info(f"{len(all_dirs)} dirs found")

        multiprocess_start = time()
        print("Start multiprocess")

        with Pool(processes=12) as p:
            p.map(spawn_child_process_to_read_docs, args_list)

        logger.info("Getting all documents from disk took: {0}".format(time() - multiprocess_start))
        logger.info(f"Saving collection docid2pos found to {id2pos_fn}...")
        with open(id2pos_fn, "w") as fp:
            json.dump(shared_id2pos.copy(), fp)

    def get_opened_file(self, fn):
        if not hasattr(self, "file_buffer"):
            self.file_buffer = {}

        if fn in self.file_buffer:
            return self.file_buffer[fn]

        if len(self.file_buffer) == self.BUFFER_SIZE:  # close the file with oldest use time
            del self.file_buffer[random.choice(list(self.file_buffer.keys()))]

        self.file_buffer[fn] = gzip.open(fn)
        return self.file_buffer[fn]

    def build(self):
        self.cleaner = Cleaner(
            comments=True,  # True = remove comments
            meta=True,  # True = remove meta tags
            scripts=True,  # True = remove script tags
            embedded=True,  # True = remove embeded tags
        )
        # clean_dom = cleaner.clean_html(original_content)

    @staticmethod
    def terms_in_line(line):
        for term in [DOC, DOCHDR, DOCNO, TERMINATING_DOC, TERMINATING_DOCHDR, TERMINATING_DOCNO]:
            if term.lower() in line.lower():
                return True
        return False

    def get_doc(self, docid):
        """ docid: in format of GX000-10-0000000 """
        dir_name, subdir, subid = docid.split("-")
        f = self.get_opened_file(f"{self.path}/{dir_name}/{subdir}.gz")
        start, offset = self.id2pos[docid]
        f.seek(start, 0)
        rawdoc = f.read(offset).decode("utf-8", errors="ignore")
        doc_lines = [line.strip() for line in clean_documents(rawdoc)]
        id, doc = parse_record(doc_lines, return_doc=True)
        assert id == docid
        try:
            doc = self.cleaner.clean_html(doc.encode())
            doc = fromstring(doc).text_content()
        except Exception as e:
            print(id, e)

        doc = " ".join(doc.split())
        return doc

# if __name__ == "__main__":
#     if sys.argv[1] == "gov":
#         benchmark = GOV2Benchmark()
#         benchmark.download_if_missing()
#
#     elif sys.argv[1] == "govsample":
#         for mode in ["deep", "shallow"]:
#             benchmark = SampledGOV2(parse_config_string(f"mode={mode} rate=1.0"))
#             benchmark.download_if_missing()
#             for rate in np.arange(0.1, 1.0, 0.2):
#                 rate = f"%.2f" % rate
#                 print(mode, rate)
#                 benchmark = SampledGOV2(parse_config_string(f"mode={mode} rate={rate}"))
#                 benchmark.download_if_missing()