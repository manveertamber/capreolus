import os
import math
import tarfile
from collections import defaultdict

import numpy as np

from capreolus import ConfigOption, Dependency, constants
from capreolus.utils.loginit import get_logger
from capreolus.utils.common import download_file
from capreolus.utils.trec import load_trec_topics

from . import Searcher


@Searcher.register
class MsmarcoPsg(Searcher):
    module_name = "msmarcopsg_searcher"
    dependencies = [Dependency(key="benchmark", module="benchmark", name="msmarcopsg")]

    @staticmethod
    def convert_to_trec_runs(msmarco_top1k_fn):
        runs = defaultdict(dict)
        with open(msmarco_top1k_fn, "r", encoding="utf-8") as f:
            for line in f:
                qid, pid, _, _ = line.strip().split("\t")
                runs[qid][pid] = len(runs.get(qid, []))
        return runs

    def download_and_extract(self, url):
        tmp_dir = self.get_cache_path() / "tmp"
        gz_name = url.split("/")[-1]
        gz_fn, extracted_fn = tmp_dir / gz_name, tmp_dir / gz_name.replace(".tar.gz", "")
        if extracted_fn.exists():
            return extracted_fn

        if not gz_fn.exists():
            tmp_dir.mkdir(exist_ok=True, parents=True)
            download_file(url, gz_fn)
        with tarfile.open(gz_fn, "r:gz") as f:
            f.extractall(path=tmp_dir)

        return extracted_fn

    def _query_from_file(self, topicsfn, output_path, cfg):
        """ only query results in dev and test set are saved """
        donefn = os.path.join(output_path, "done")
        if os.path.exists(donefn):
            return output_path

        qids = set(load_trec_topics(topicsfn)["title"].keys())

        allruns = {}
        urls = [
            "https://msmarco.blob.core.windows.net/msmarcoranking/top1000.dev.tar.gz",
            "https://msmarco.blob.core.windows.net/msmarcoranking/top1000.eval.tar.gz"]

        for url in urls:
            extract_fn = self.download_and_extract(url)
            runs = self.convert_to_trec_runs(extract_fn)
            allruns.update(runs)

        assert len(set(allruns.keys()) - qids) == 0

        output_path.mkdir(exist_ok=True, parents=True)
        self.write_trec_run(preds=runs, outfn=(output_path / "searcher"))
        with open(donefn, "wt") as donef:
            print("done", file=donef)
        return output_path
