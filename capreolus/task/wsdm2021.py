from pathlib import Path
from collections import defaultdict

from capreolus.task import Task, RerankTask
from capreolus.sampler import TrainTripletSampler, PredSampler
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)


@Task.register
class WSDM2021(RerankTask):
    module_name = "wsdm2021"

    def rerank_run(self, best_search_run, train_output_path, include_train=False):
        if not isinstance(train_output_path, Path):
            train_output_path = Path(train_output_path)

        fold = self.config["fold"]
        threshold = self.config["threshold"]
        dev_output_path = train_output_path / "pred" / "dev"
        logger.debug("results path: %s", train_output_path)

        docids = set(docid for querydocs in best_search_run.values() for docid in querydocs)
        self.reranker.extractor.preprocess(
            qids=best_search_run.keys(), docids=docids, topics=self.benchmark.topics[self.benchmark.query_type]
        )
        self.reranker.build_model()
        self.reranker.searcher_scores = best_search_run

        train_run = {qid: docs for qid, docs in best_search_run.items() if
                     qid in self.benchmark.folds[fold]["train_qids"]}
        # For each qid, select the top 100 (defined by config["threshold") docs to be used in validation
        dev_run = defaultdict(dict)
        # This is possible because best_search_run is an OrderedDict
        for qid, docs in best_search_run.items():
            if qid in self.benchmark.folds[fold]["predict"]["dev"]:
                for idx, (docid, score) in enumerate(docs.items()):
                    if idx >= threshold:
                        break
                    dev_run[qid][docid] = score

        # Depending on the sampler chosen, the dataset may generate triplets or pairs
        train_dataset = self.sampler
        train_dataset.prepare(
            train_run, self.benchmark.sampled_qrels, self.reranker.extractor,
            relevance_level=self.benchmark.relevance_level,
        )
        dev_dataset = PredSampler()
        dev_dataset.prepare(
            dev_run, self.benchmark.sampled_qrels, self.reranker.extractor, relevance_level=self.benchmark.relevance_level,
        )

        self.reranker.trainer.train(
            self.reranker,
            train_dataset,
            train_output_path,
            dev_dataset,
            dev_output_path,
            self.benchmark.sampled_qrels,
            self.config["optimize"],
            self.benchmark.relevance_level,
        )

        self.reranker.trainer.load_best_model(self.reranker, train_output_path)
        dev_output_path = train_output_path / "pred" / "dev" / "best"
        dev_preds = self.reranker.trainer.predict(self.reranker, dev_dataset, dev_output_path)

        test_run = defaultdict(dict)
        # This is possible because best_search_run is an OrderedDict
        for qid, docs in best_search_run.items():
            if qid in self.benchmark.folds[fold]["predict"]["test"]:
                for idx, (docid, score) in enumerate(docs.items()):
                    if idx >= threshold:
                        break
                    test_run[qid][docid] = score

        test_dataset = PredSampler()
        test_dataset.prepare(
            test_run, self.benchmark.qrels, self.reranker.extractor, relevance_level=self.benchmark.relevance_level
        )
        test_output_path = train_output_path / "pred" / "test" / "best"
        test_preds = self.reranker.trainer.predict(self.reranker, test_dataset, test_output_path)

        preds = {"dev": dev_preds, "test": test_preds}

        if include_train:
            train_dataset = PredSampler(
                train_run, self.benchmark.sampled_qrels, self.reranker.extractor,
                relevance_level=self.benchmark.relevance_level,
            )

            train_output_path = train_output_path / "pred" / "train" / "best"
            train_preds = self.reranker.trainer.predict(self.reranker, train_dataset, train_output_path)
            preds["train"] = train_preds

        return preds
