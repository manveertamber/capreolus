from profane import ConfigOption, Dependency

from capreolus import evaluator
from capreolus.sampler import PredDataset
from capreolus.searcher import Searcher
from capreolus.task import Task
from capreolus.utils.loginit import get_logger
from .rank import RankTask

logger = get_logger(__name__)


@Task.register
class FilterRankTask(RankTask):
    module_name = "filterrank"
    # use the commands, searcheval(), evaluate() from RankTask
    config_spec = [
        ConfigOption("fold", "s1", "fold to run"),
        ConfigOption("optimize", "mrr", "metric to maximize on the dev set"),
        ConfigOption("metrics", ["mrr", "map", "judged_10", "P_5", "ndcg_cut_10"], "metrics reported for evaluation", value_type="strlist"),
    ]
    dependencies = [
        Dependency(key="benchmark", module="benchmark", name="codesearchnet_corpus", provide_this=True, provide_children=["collection"]),
        Dependency(key="rank", module="task", name="rank"),   # take the benchmark
        Dependency(key="searcher", module="searcher", name="BM25")
    ]

    def search(self):
        fold, metric = self.config["fold"], self.config["optimize"]

        self.rank.search()
        rank_results = self.rank.evaluate()
        best_search_run_path = rank_results["path"][fold]

        self.searcher.index.create_index()
        if self.searcher.module_name == "BM25_reranker":
            best_search_run = Searcher.load_trec_run(best_search_run_path)
            search_results_folder = self.searcher.query_from_file(
                self.benchmark.topic_file, self.get_results_path(), best_search_run)
        elif self.searcher.module_name == "BM25RM3":
            search_results_folder = self.searcher.query_from_file(
                self.benchmark.topic_file, self.get_results_path(), rerank=True, run_fn=best_search_run_path)
        else:
            raise ValueError(f"Unsupported seearcher: {self.searcher.name}")

        print("Search rerank results are at: " + str(search_results_folder))
