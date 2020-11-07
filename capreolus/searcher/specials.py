import os
from time import time
from collections import defaultdict
from pathlib import Path

from capreolus import ConfigOption, Dependency, constants
from capreolus.utils.loginit import get_logger
from capreolus.utils.trec import load_trec_topics

from . import Searcher

logger = get_logger(__name__)


@Searcher.register
class MsmarcoPsg(Searcher):
    module_name = "msmarcopsg"
    dependencies = [Dependency(key="benchmark", module="benchmark", name="msmarcopsg")]

    @staticmethod
    def convert_to_trec_runs(msmarco_top1k_fn, train):
        runs = defaultdict(dict)
        with open(msmarco_top1k_fn, "r", encoding="utf-8") as f:
            for line in f:
               if train:
                    qid, pos_pid, neg_pid = line.strip().split("\t")
                    runs[qid][pos_pid] = len(runs.get(pos_pid, []))
                    runs[qid][neg_pid] = len(runs.get(neg_pid, []))
               else:
                    qid, pid, _, _ = line.strip().split("\t")
                    runs[qid][pid] = len(runs.get(qid, []))
        return runs

    def _query_from_file(self, topicsfn, output_path, cfg):
        """ only query results in dev and test set are saved """
        donefn = os.path.join(output_path, "done")
        if os.path.exists(donefn):
            return output_path

        qids = set(load_trec_topics(topicsfn)["title"].keys())

        urls = [
            "https://msmarco.blob.core.windows.net/msmarcoranking/qidpidtriples.train.full.tsv.gz",
            "https://msmarco.blob.core.windows.net/msmarcoranking/top1000.dev.tar.gz",
            # "https://msmarco.blob.core.windows.net/msmarcoranking/top1000.eval.tar.gz"
        ]

        tmp_dir = self.get_cache_path() / "tmp"
        output_path.mkdir(exist_ok=True, parents=True)
        left_qids = set(qids)
        mode = "wt"
        for url in urls:
            extract_file_name = url.split("/")[-1].replace(".gz", "").replace(".tar", "")
            if ".dev" in extract_file_name: 
                extract_dir = Path("/GW/carpet/nobackup/czhang/anserini/runs")  
                extract_file_name = "run.msmarco-passage.dev.small.trec"
                runs = self.load_trec_run(extract_dir / extract_file_name)
            else:
                extract_dir = self.benchmark.collection.download_and_extract(url, tmp_dir, expected_fns=extract_file_name)
                runs = self.convert_to_trec_runs(extract_dir / extract_file_name, train=(".train." in url))
            t = time()
            left_qids = left_qids - set(runs.keys()) 
            self.write_trec_run(preds=runs , outfn=(output_path / "searcher"), mode=mode)
            logger.info(f"writing runs to file... left {len(left_qids)} takes {time() - t} sec.")
            mode = "a"

        # assert len(set(allruns.keys()) - qids) == 0
        # assert len(left_qids) == 0
        if len(left_qids) != 0:
            logger.warning(f"There are unfound qids, {len(left_qids)}")
        # self.write_trec_run(preds=allruns, outfn=(output_path / "searcher"))
        # logger.info(f"Wrote runs into to {output_path}/searcher, time: {time() - t}")

        with open(donefn, "wt") as donef:
            print("done", file=donef)
        return output_path

