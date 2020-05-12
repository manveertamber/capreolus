import os
from capreolus.task import Task
from capreolus.registry import RESULTS_BASE_PATH

from capreolus import evaluator


def describe(config, modules):
    output_path = _pipeline_path(config, modules)
    return Task.describe_pipeline(config, modules, output_path)


def train(config, modules):
    fold, metric = config["fold"], config["optimize"]

    base_searcher = modules["searcher"]["searcher"]
    base_benchmark = base_searcher["benchmark"]
    rerank_searcher = modules["searcher"]
    rerank_benchmark = modules["benchmark"]

    base_path = base_searcher.get_cache_path() / base_benchmark.name
    rerank_path = rerank_searcher.get_cache_path() / rerank_benchmark.name

    print("base path: ", base_path)
    if os.path.exists(os.path.join(rerank_path, "done")):
        print(rerank_path)
        print("Done Training")
        # return

    # base searcher
    topics_fn = base_benchmark.topic_file
    search_results_folder = base_searcher.query_from_file(topics_fn, base_path)
    print("Search base results are at: " + search_results_folder)

    # load runs of base searcher
    best_results_pathes = evaluator.search_best_run(
        base_path, base_benchmark, primary_metric=metric, metrics=[metric])["path"][fold]
    best_runs = base_searcher.load_trec_run(best_results_pathes)

    # best_runs = {"408747": ["python-FUNCTION-448595"]}  # tmp
    # reranker searcher
    rerank_searcher["index"].create_index()
    search_results_folder = rerank_searcher.query_from_file(topics_fn, rerank_path, best_runs)
    print("Search rerank results are at: " + str(search_results_folder))


def evaluate(config, modules):
    searcher = modules["searcher"]
    benchmark = modules["benchmark"]

    metric = config["optimize"]
    all_metric = ["mrr", "ndcg_cut_20", "ndcg_cut_10", "map", "P_20", "P_10", "set_recall"]
    output_dir = searcher.get_cache_path() / benchmark.name
    print("output_dir: ", output_dir)

    best_results = evaluator.search_best_run(output_dir, benchmark, primary_metric=metric, metrics=all_metric)
    print(best_results, "***")

    pathes = [f"\t{s}: {path}" for s, path in best_results["path"].items()]
    print("path for each split: \n", "\n".join(pathes))

    scores = [f"\t{s}: {score}" for s, score in best_results["score"].items()]
    print(f"cross-validated results when optimizing for {metric}: \n", "\n".join(scores))


def _pipeline_path(config, modules):
    pipeline_cfg = {k: v for k, v in config.items() if k not in modules and k not in ["expid"]}
    pipeline_path = "_".join(["task-rank"] + [f"{k}-{v}" for k, v in sorted(pipeline_cfg.items())])
    output_path = (
        RESULTS_BASE_PATH
        / config["expid"]
        / modules["collection"].get_module_path()
        / modules["searcher"].get_module_path(include_provided=False)
        / pipeline_path
        / modules["benchmark"].get_module_path()
    )

    return output_path


class RankTask(Task):
    def pipeline_config():
        expid = "debug"
        seed = 123_456
        fold = "s1"
        # eval_metrics = {"map", "ndcg_cut_20", "ndcg_cut_10", "P_20"}
        optimize = "map"  # metric to maximize on the dev set

    name = "nonnn_rerank"
    module_order = ["collection", "searcher", "benchmark"]
    module_defaults = {
        "searcher": "BM25_reranker",
        "collection": "codesearchnet",
        "benchmark": "codesearchnet_corpus"}
    config_functions = [pipeline_config]
    config_overrides = []
    commands = {"train": train, "evaluate": evaluate, "describe": describe}
    default_command = "describe"
