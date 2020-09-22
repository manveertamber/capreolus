import json
from capreolus import Dependency, constants
from capreolus.utils.trec import topic_to_trectxt

from . import Benchmark

PACKAGE_PATH = constants["PACKAGE_PATH"]


@Benchmark.register
class TRECCAR(Benchmark):
    """Robust04 benchmark using the title folds from Huston and Croft. [1] Each of these is used as the test set.
    Given the remaining four folds, we split them into the same train and dev sets used in recent work. [2]

    [1] Samuel Huston and W. Bruce Croft. 2014. Parameters learned in the comparison of retrieval models using term dependencies. Technical Report.

    [2] Sean MacAvaney, Andrew Yates, Arman Cohan, Nazli Goharian. 2019. CEDR: Contextualized Embeddings for Document Ranking. SIGIR 2019.
    """
    module_name = "treccar"
    dependencies = [Dependency(key="collection", module="collection", name="treccar")]
    file_dir = PACKAGE_PATH / "data" / "treccar"
    qrel_file = file_dir / "qrels.treccar.txt"
    topic_file = file_dir / "topics.treccar.txt"
    fold_file = file_dir / "rob04_cedr_folds.json"
    query_type = "title"

    def build(self):
        self.download_if_missing()

    @staticmethod
    def clean_topic(topic_str):
        return topic_str.replace("enwiki:", "").replace("%20", " ").replace("/", " ").strip()

    def download_if_missing(self):
        if all([f.exists() for f in [self.fold_file, self.qrel_file, self.topic_file]]):
            return

        self.file_dir.mkdir(exist_ok=True, parents=True)
        """use the ones provided by monobert first """
        monobert_dir = self.get_cache_path() / "tmp" / "monobert_treccar"
        monobert_dir.mkdir(exist_ok=True, parents=True)
        print("benchmark", monobert_dir)
        # exit(0)

        topic2id_file = self.file_dir / "topic2id_file.treccar.json"
        qrel_f, topic_f = open(self.qrel_file, "w"), open(self.topic_file, "w")

        topic2ori, folds = {}, {setname: [] for setname in ["train", "dev", "test"]}
        for setname in folds:
            with open(monobert_dir / f"{setname}.topics") as f:
                for ori in f:
                    qid, ori, cleaned = len(topic2ori), ori.strip(), self.clean_topic(ori)

                    topic2ori[ori] = qid
                    folds[setname].append(qid)
                    topic_f.write(topic_to_trectxt(qid, cleaned))

            with open(monobert_dir / f"{setname}.qrels") as f:
                for line in f:
                    query, _, docid, label = line.strip().split()
                    qid = topic2ori[query]
                    qrel_f.write(f"{qid} 0 {docid} {label}\n")

        json.dump(
            {"s1": {"train_qids": folds["train"], "predict": {"dev": folds["dev"], "test":folds["test"]}}},
            open(self.fold_file, "w"))
        json.dump(topic2ori, open(topic2id_file, "w"))
