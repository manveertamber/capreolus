import os
import json
import glob
import gzip
import math
import subprocess
from collections import defaultdict, OrderedDict

from tqdm import tqdm
import numpy as np

from pyserini.search import pysearch
from capreolus.registry import ModuleBase, RegisterableModule, Dependency, MAX_THREADS, PACKAGE_PATH
from capreolus.utils.common import Anserini
from capreolus.utils.trec import load_trec_topics
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)  # pylint: disable=invalid-name


class Searcher(ModuleBase, metaclass=RegisterableModule):
    """the module base class"""

    module_type = "searcher"

    @staticmethod
    def load_trec_run(fn):
        run = defaultdict(dict)
        with open(fn, "rt") as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    qid, _, docid, rank, score, desc = line.split(" ")
                    run[qid][docid] = float(score)
        return run

    @staticmethod
    def write_trec_run(preds, outfn):
        count = 0
        with open(outfn, "wt") as outf:
            for qid in sorted(preds):
                rank = 1
                for docid, score in sorted(preds[qid].items(), key=lambda x: x[1], reverse=True):
                    print(f"{qid} Q0 {docid} {rank} {score} capreolus", file=outf)
                    rank += 1
                    count += 1


class AnseriniSearcherMixIn:
    """ MixIn for searchers that use Anserini's SearchCollection script """

    def _anserini_query_from_file(self, topicsfn, anserini_param_str, output_base_path):
        if not os.path.exists(topicsfn):
            raise IOError(f"could not find topics file: {topicsfn}")

        donefn = os.path.join(output_base_path, "done")
        if os.path.exists(donefn):
            logger.debug(f"skipping Anserini SearchCollection call because path already exists: {donefn}")
            return

        # create index if it does not exist. the call returns immediately if the index does exist.
        self["index"].create_index()

        os.makedirs(output_base_path, exist_ok=True)
        output_path = os.path.join(output_base_path, "searcher")

        # add stemmer and stop options to match underlying index
        indexopts = f"-stemmer {self['index'].cfg['stemmer']}"
        if self["index"].cfg["indexstops"]:
            indexopts += " -keepstopwords"

        index_path = self["index"].get_index_path()
        anserini_fat_jar = Anserini.get_fat_jar()
        cmd = f"java -classpath {anserini_fat_jar} -Xms512M -Xmx31G -Dapp.name=SearchCollection io.anserini.search.SearchCollection -topicreader Trec -index {index_path} {indexopts} -topics {topicsfn} -output {output_path} -inmem -threads {MAX_THREADS} {anserini_param_str}"
        logger.info("Anserini writing runs to %s", output_path)
        logger.debug(cmd)

        app = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)

        # Anserini output is verbose, so ignore DEBUG log lines and send other output through our logger
        for line in app.stdout:
            Anserini.filter_and_log_anserini_output(line, logger)

        app.wait()
        if app.returncode != 0:
            raise RuntimeError("command failed")

        with open(donefn, "wt") as donef:
            print("done", file=donef)


class BM25(Searcher, AnseriniSearcherMixIn):
    """ BM25 with fixed k1 and b. """

    name = "BM25"
    dependencies = {"index": Dependency(module="index", name="anserini")}

    @staticmethod
    def config():
        b = 0.4  # controls document length normalization
        k1 = 0.9  # controls term saturation
        hits = 1000

    def query_from_file(self, topicsfn, output_path):
        """
        Runs BM25 search. Takes a query from the topic files, and fires it against the index
        Args:
            topicsfn: Path to a topics file
            output_path: Path where the results of the search (i.e the run file) should be stored

        Returns: Path to the run file where the results of the search are stored

        """
        bs = [self.cfg["b"]]
        k1s = [self.cfg["k1"]]
        bstr = " ".join(str(x) for x in bs)
        k1str = " ".join(str(x) for x in k1s)
        hits = self.cfg["hits"]
        anserini_param_str = f"-bm25 -bm25.b {bstr} -bm25.k1 {k1str} -hits {hits}"
        self._anserini_query_from_file(topicsfn, anserini_param_str, output_path)

        return output_path

    def query(self, query):
        self["index"].create_index()
        searcher = pysearch.SimpleSearcher(self["index"].get_index_path().as_posix())
        searcher.set_bm25_similarity(self.cfg["k1"], self.cfg["b"])

        hits = searcher.search(query)
        return OrderedDict({hit.docid: hit.score for hit in hits})


class BM25Grid(Searcher, AnseriniSearcherMixIn):
    """ BM25 with a grid search for k1 and b. Search is from 0.1 to bmax/k1max in 0.1 increments """

    name = "BM25Grid"
    dependencies = {"index": Dependency(module="index", name="anserini")}

    @staticmethod
    def config():
        k1max = 1.0  # maximum k1 value to include in grid search
        bmax = 1.0  # maximum b value to include in grid search
        k1min = 0.1  # minimum k1 value to include in grid search
        bmin = 0.1  # minimum b value to include in grid search
        hits = 1000

    def query_from_file(self, topicsfn, output_path):
        bs = np.around(np.arange(self.cfg["bmin"], self.cfg["bmax"] + 0.1, 0.1), 1)
        k1s = np.around(np.arange(self.cfg["k1min"], self.cfg["k1max"] + 0.1, 0.1), 1)

        bstr = " ".join(str(x) for x in bs)
        k1str = " ".join(str(x) for x in k1s)
        hits = self.cfg["hits"]
        anserini_param_str = f"-bm25 -bm25.b {bstr} -bm25.k1 {k1str} -hits {hits}"

        self._anserini_query_from_file(topicsfn, anserini_param_str, output_path)

        return output_path

    def query(self, query, b, k1):
        self["index"].create_index()
        searcher = pysearch.SimpleSearcher(self["index"].get_index_path().as_posix())
        searcher.set_bm25_similarity(k1, b)

        hits = searcher.search(query)
        return OrderedDict({hit.docid: hit.score for hit in hits})


class BM25RM3(Searcher, AnseriniSearcherMixIn):

    name = "BM25RM3"
    dependencies = {"index": Dependency(module="index", name="anserini")}

    @staticmethod
    def config():
        k1 = BM25RM3.list2str([0.65, 0.70, 0.75])
        b = BM25RM3.list2str([0.60, 0.7])  # [0.60, 0.65, 0.7]
        fbTerms = BM25RM3.list2str([65, 70, 95, 100])
        fbDocs = BM25RM3.list2str([5, 10, 15])
        originalQueryWeight = BM25RM3.list2str([0.2, 0.25])
        hits = 1000

    @staticmethod
    def list2str(l):
        return "-".join(str(x) for x in l)

    def query_from_file(self, topicsfn, output_path):
        paras = {k: " ".join(self.cfg[k].split("-")) for k in ["k1", "b", "fbTerms", "fbDocs", "originalQueryWeight"]}
        hits = str(self.cfg["hits"])

        anserini_param_str = (
            "-rm3 "
            + " ".join(f"-rm3.{k} {paras[k]}" for k in ["fbTerms", "fbDocs", "originalQueryWeight"])
            + " -bm25 "
            + " ".join(f"-bm25.{k} {paras[k]}" for k in ["k1", "b"])
            + f" -hits {hits}"
        )
        self._anserini_query_from_file(topicsfn, anserini_param_str, output_path)

        return output_path

    def query(self, query, b, k1, fbterms, fbdocs, ow):
        self["index"].create_index()
        searcher = pysearch.SimpleSearcher(self["index"].get_index_path().as_posix())
        searcher.set_bm25_similarity(k1, b)
        searcher.set_rm3_reranker(fb_terms=fbterms, fb_docs=fbdocs, original_query_weight=ow)

        hits = searcher.search(query)
        return OrderedDict({hit.docid: hit.score for hit in hits})


class BM25Reranker(Searcher):
    name = "BM25_reranker"
    dependencies = {
        "index": Dependency(module="index", name="anserini_tf"),
        "searcher": Dependency(module="searcher", name="csn_distractors"),
    }

    @staticmethod
    def config():
        b = 0.4
        k1 = 0.9
        hits = 1000

    def __calc_bm25(self, query, docid):
        k1, b = self.cfg["k1"], self.cfg["b"]
        avg_doc_len = self["index"].get_avglen()
        doclen = self["index"].get_doclen(docid)

        # print(f">>>>>> {query} {docid} avg doc len: ", avg_doc_len, "doclen: ", doclen)

        if doclen == -1:
            return -math.inf

        def term_bm25(term):
            tf = self["index"].get_tf(term, docid)
            if math.isnan(tf):  # a == math.nan will always be False!
                return math.nan

            idf = self["index"].get_idf(term)
            numerator = tf  # * (k1 + 1)  # ignore the (k1+1) since the constant does not affect the ranking result
            denominator = tf + k1 * (1 - b + b * doclen / avg_doc_len)
            # print(term, "idf: ", idf, "tf: ", tf, "bm25: ", idf * numerator / denominator)
            return idf * numerator / denominator

        bm25_per_qterm = [term_bm25(qterm) for qterm in query]
        # print(dict(zip(query, bm25_per_qterm)))
        # print(np.nansum(bm25_per_qterm))
        return np.nansum(bm25_per_qterm)

    def calc_bm25(self, query, docids):
        bm25 = {docid: self.__calc_bm25(query, docid) for docid in docids}
        bm25 = sorted(bm25.items(), key=lambda k_v: k_v[1], reverse=True)
        bm25 = {docid: bm25 for i, (docid, bm25) in enumerate(bm25) if i < self.cfg["hits"]}
        return bm25

    def query_from_file(self, topicsfn, output_path, runs=None):
        """ only perform bm25 on the docs in runs """
        donefn = os.path.join(output_path, "done")
        if os.path.exists(donefn):
            logger.debug(f"done file for {self.name} already exists, skip search")
            # return output_path

        self["index"].open()

        # prepare topic
        cache_fn = self.get_cache_path()
        cache_fn.mkdir(exist_ok=True, parents=True)

        # topics = load_trec_topics(topicsfn)["title"]
        # print(409130, topics["408747"])
        # print(cache_fn)
        # print("after analyze: ")
        # print(self["index"].analyze_sent(topics["408747"]))
        topic_cache_path = cache_fn / "topic.analyze.json"
        if os.path.exists(topic_cache_path):
            topics = json.load(open(topic_cache_path))
            print(f"loading analyzed topic from cache {topic_cache_path}")
        else:
            topics = load_trec_topics(topicsfn)["title"]
            topics = {qid: self["index"].analyze_sent(q) for qid, q in tqdm(topics.items(), desc="Transforming query")}
            json.dump(topics, open(topic_cache_path, "w"))
            print(f"storing analyzed topic from cache {topic_cache_path}")

        # print("from loaded: ")
        # print(topics["408747"])

        bm25runs = {}
        docnos = self["index"]["collection"].get_docnos()

        # topics = {"408747": topics["408747"]}
        for qid, query in tqdm(topics.items(), desc=f"Calculating bm25"):
            # print(qid, query)
            docids = runs.get(qid, None) if runs else docnos
            bm25runs[qid] = self.calc_bm25(query, docids) if docids else {}
        # raise

        os.makedirs(output_path, exist_ok=True)
        self.write_trec_run(bm25runs, os.path.join(output_path, "searcher"))

        with open(donefn, "wt") as donef:
            print("done", file=donef)

        return output_path


class StaticBM25RM3Rob04Yang19(Searcher):
    """ Tuned BM25+RM3 run used by Yang et al. in [1]. This should be used only with a benchmark using the same folds and queries.

        [1] Wei Yang, Kuang Lu, Peilin Yang, and Jimmy Lin. Critically Examining the "Neural Hype": Weak Baselines and  the Additivity of Effectiveness Gains from Neural Ranking Models. SIGIR 2019.
    """

    name = "bm25staticrob04yang19"

    def query_from_file(self, topicsfn, output_path):
        import shutil

        outfn = os.path.join(output_path, "static.run")
        os.makedirs(output_path, exist_ok=True)
        shutil.copy2(PACKAGE_PATH / "data" / "rob04_yang19_rm3.run", outfn)

        return output_path

    def query(self, *args, **kwargs):
        raise NotImplementedError("this searcher uses a static run file, so it cannot handle new queries")


class DirichletQL(Searcher, AnseriniSearcherMixIn):
    """ Dirichlet QL with a fixed mu """

    name = "DirichletQL"
    dependencies = {"index": Dependency(module="index", name="anserini")}

    @staticmethod
    def config():
        mu = 1000  # mu smoothing parameter
        hits = 1000

    def query_from_file(self, topicsfn, output_path):
        """
        Runs Dirichlet QL search. Takes a query from the topic files, and fires it against the index
        Args:
            topicsfn: Path to a topics file
            output_path: Path where the results of the search (i.e the run file) should be stored

        Returns: Path to the run file where the results of the search are stored

        """
        mus = [self.cfg["mu"]]
        mustr = " ".join(str(x) for x in mus)
        hits = self.cfg["hits"]
        anserini_param_str = f"-qld -mu {mustr} -hits {hits}"
        self._anserini_query_from_file(topicsfn, anserini_param_str, output_path)

        return output_path

    def query(self, query):
        self["index"].create_index()
        searcher = pysearch.SimpleSearcher(self["index"].get_index_path().as_posix())
        searcher.set_lm_dirichlet_similarity(self.cfg["mu"])

        hits = searcher.search(query)
        return OrderedDict({hit.docid: hit.score for hit in hits})


class CodeSearchDistractor(Searcher):
    """ Providing the 999 distractor documents """

    name = "csn_distractors"
    dependencies = {"benchmark": Dependency(module="benchmark", name="codesearchnet_corpus")}

    def query_from_file(self, topicsfn, output_path):
        donefn = os.path.join(output_path, "done")
        if os.path.exists(donefn):
            logger.debug(f"done file for {self.name} already exists, skip search")
            return str(output_path)

        benchmark = self["benchmark"]
        lang = benchmark.cfg["lang"]

        csn_rawdata_dir, _ = benchmark.download_raw_data()
        csn_lang_dir = os.path.join(csn_rawdata_dir, lang, "final", "jsonl")

        runs = defaultdict(dict)
        # for set_name in ["train", "valid", "test"]:
        for set_name in ["test"]:
            csn_lang_path = os.path.join(csn_lang_dir, set_name)
            neighbour_size = 20 if set_name == "train" else 1000

            objs = []
            for fn in glob.glob(os.path.join(csn_lang_path, "*.jsonl.gz")):
                with gzip.open(fn, "rb") as f:
                    lines = f.readlines()
                    for line in tqdm(lines, desc=f"Processing set {set_name} {os.path.basename(fn)}"):
                        objs.append(json.loads(line))

                        if len(objs) == neighbour_size:  # 1 ground truth and 999 distractor docs
                            for obj1 in objs:
                                qid = benchmark.get_qid(obj1["docstring_tokens"], parse=True)
                                gt_docid = benchmark.get_docid(obj1["url"], obj1["code_tokens"], parse=True)
                                all_docs = []

                                for rank, obj2 in enumerate(objs):
                                    docid = benchmark.get_docid(obj2["url"], obj2["code_tokens"], parse=True)
                                    all_docs.append(docid)
                                    runs[qid][docid] = 1.0 / (rank + 1)
                                assert gt_docid in all_docs
                            objs = []  # reset

            # for valid and test set, in case of duplicated qid: preserve the only 1k neighbors
            if set_name == "train":
                continue

            for qid in runs:
                if len(runs[qid]) == neighbour_size:
                    continue
                top1kneighbor = sorted(runs[qid].items(), key=lambda k_v: k_v[1])
                runs[qid] = {k: v for k, v in top1kneighbor[-neighbour_size:]}

        os.makedirs(output_path, exist_ok=True)
        self.write_trec_run(runs, os.path.join(output_path, "searcher"))

        with open(donefn, "wt") as donef:
            print("done", file=donef)
        return str(output_path)

    def query(self, *args, **kwargs):
        raise NotImplementedError("this searcher uses a static run file, so it cannot handle new queries")
