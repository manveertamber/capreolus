import hashlib
from copy import deepcopy
from time import time

import torch.utils.data

from profane import ModuleBase, Dependency, ConfigOption, constants
from capreolus.utils.exceptions import MissingDocError
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)


class Sampler(ModuleBase):
    module_type = "sampler"
    requires_random_seed = True

    def prepare(self, runs, qids, qrels, extractor, relevance_level=1, **kwargs):
        """
        params:
        qid_to_docids: A dict of the form {qid: [list of docids to rank]}
        qrels: A dict of the form {qid: {docid: label}}
        extractor: An Extractor instance (eg: EmbedText)
        relevance_level: Threshold score below which documents are considered to be non-relevant.
        """
        self.extractor = extractor
        self.runs = runs  # Note, delete this runs would affect other sampler/module using same runs!

        t1 = time()
        missed_qids = [qid for qid in runs if qid not in qrels]
        if missed_qids:
            warning_msg = \
                "skipping qids that were missing from the qrels: {}".format(missed_qids) if len(missed_qids) < 10 else \
                "skipping qids that were missing from the qrels: {} qids in totoal".format(len(missed_qids))
            logger.warning(warning_msg)
        print("found missed qid: ", time()-t1)
        t1 = time()

        # remove qids not exists in all of {runs, qids, qrels}
        self.qid_to_reldocs = {
            qid: [docid for docid in qrels[qid] if qrels[qid][docid] >= relevance_level]
            for qid in runs if (qid in qids and qid in qrels)
        }
        print("found qid2reldocs: ", time()-t1)
        self.total_samples = 0
        t1 = time()
        self.clean()
        print(f"after cleaning: ", time() - t1)

    def get_pos_neg_pair(self, qid):
        alldocs = self.runs[qid]
        all_posdocs = self.qid_to_reldocs[qid]
        all_negdocs = list(set(alldocs) - set(all_posdocs))
        return self.rng.choice(all_posdocs), self.rng.choice(all_negdocs)

    def clean(self):
        # remove any ids that do not have any relevant docs or any non-relevant docs for training
        total_samples = 0  # keep tracks of the total possible number of unique training triples for this dataset
        for qid, doc2score in self.runs.items():
            if qid not in self.qid_to_reldocs:
                continue

            posdocs = [docid for docid in doc2score if docid in self.qid_to_reldocs[qid]]
            n_posdocs = len(posdocs)
            n_negdocs = len(doc2score) - n_posdocs
            if n_posdocs == 0 or n_negdocs == 0:
                logger.debug(
                    "removing training qid=%s with %s positive docs and %s negative docs", qid, n_posdocs, n_negdocs)
                del self.qid_to_reldocs[qid]
            else:
                if set(self.qid_to_reldocs[qid]) != set(posdocs):  # remove the pos docs not appeared in the runs
                    self.qid_to_reldocs[qid] = posdocs
                total_samples += len(posdocs) * n_negdocs
        self.total_samples = total_samples

    def get_hash(self):
        raise NotImplementedError

    def get_total_samples(self):
        return self.total_samples

    def generate_samples(self):
        raise NotImplementedError


@Sampler.register
class TrainTripletSampler(Sampler, torch.utils.data.IterableDataset):
    """
    Samples training data triplets. Each samples is of the form (query, relevant doc, non-relevant doc)
    """

    module_name = "triplet"

    def __hash__(self):
        return self.get_hash()

    def get_hash(self):
        # TODO: shrink the size of sorted_rep
        sorted_rep = sorted([(qid, list(docids)) for qid, docids in self.runs.items()])
        key_content = "{0}{1}".format(self.extractor.get_cache_path(), str(sorted_rep))
        key = hashlib.md5(key_content.encode("utf-8")).hexdigest()
        return "triplet_{0}".format(key)

    def generate_samples(self):
        """
        Generates triplets infinitely.
        """
        all_qids = sorted(self.qid_to_reldocs)
        if len(all_qids) == 0:
            raise RuntimeError("TrainDataset has no valid qids")

        while True:
            self.rng.shuffle(all_qids)

            for qid in all_qids:
                posdocid, negdocid = self.get_pos_neg_pair(qid)
                try:
                    # Convention for label - [1, 0] indicates that doc belongs to class 1 (i.e relevant
                    # ^ This is used with categorical cross entropy loss
                    yield self.extractor.id2vec(qid, posdocid, negdocid, label=[1, 0])
                except MissingDocError as e:
                    # at training time we warn but ignore on missing docs
                    logger.warning(e)
                    logger.warning(
                        "skipping training pair with missing features: qid=%s posid=%s negid=%s", qid, posdocid, negdocid
                    )

    def __iter__(self):
        """
        Returns: Triplets of the form (query_feature, posdoc_feature, negdoc_feature)
        """

        return iter(self.generate_samples())


@Sampler.register
class TrainPairSampler(Sampler, torch.utils.data.IterableDataset):
    """
    Samples training data pairs. Each sample is of the form (query, doc)
    The number of generate positive and negative samples are the same.
    """

    module_name = "pair"
    dependencies = []

    def get_hash(self):
        sorted_rep = sorted([(qid, docids) for qid, docids in self.qid_to_docids.items()])
        key_content = "{0}{1}".format(self.extractor.get_cache_path(), str(sorted_rep))
        key = hashlib.md5(key_content.encode("utf-8")).hexdigest()
        return "pair_{0}".format(key)

    def generate_samples(self):
        all_qids = sorted(self.qid_to_reldocs)
        if len(all_qids) == 0:
            raise RuntimeError("TrainDataset has no valid training pairs")

        while True:
            self.rng.shuffle(all_qids)
            for qid in all_qids:
                # Convention for label - [1, 0] indicates that doc belongs to class 1 (i.e relevant
                # ^ This is used with categorical cross entropy loss
                posdocid, negdocid = self.get_pos_neg_pair(qid)
                yield self.extractor.id2vec(qid, posdocid, negid=None, label=[0, 1])
                yield self.extractor.id2vec(qid, negdocid, negid=None, label=[1, 0])

    def __iter__(self):
        return iter(self.generate_samples())


@Sampler.register
class PredSampler(Sampler, torch.utils.data.IterableDataset):
    """
    Creates a Dataset for evaluation (test) data to be used with a pytorch DataLoader
    """

    module_name = "pred"
    config_spec = [
        ConfigOption("seed", 1234),
    ]

    def get_hash(self):
        # sorted_rep = sorted([(qid, docids) for qid, docids in self.qid_to_docids.items()])
        sorted_rep = sorted([(qid, list(docids)) for qid, docids in self.runs.items() if qid in self.qid_to_reldocs])
        key_content = "{0}{1}".format(self.extractor.get_cache_path(), str(sorted_rep))
        key = hashlib.md5(key_content.encode("utf-8")).hexdigest()

        return "dev_{0}".format(key)

    def generate_samples(self):
        for qid, docids in self.runs.items():
            if qid not in self.qid_to_reldocs:
                continue

            for docid in docids:
                try:
                    if docid in self.qid_to_reldocs[qid]:
                        yield self.extractor.id2vec(qid, docid, label=[0, 1])
                    else:
                        yield self.extractor.id2vec(qid, docid, label=[1, 0])
                except MissingDocError:
                    # when predictiong we raise an exception on missing docs, as this may invalidate results
                    logger.error("got none features for prediction: qid=%s posid=%s", qid, docid)
                    # raise

    def __hash__(self):
        return self.get_hash()

    def __iter__(self):
        """
        Returns: Tuples of the form (query_feature, posdoc_feature)
        """

        return iter(self.generate_samples())

    def get_qid_docid_pairs(self):
        """
        Returns a generator for the (qid, docid) pairs. Useful if you want to sequentially access the pred pairs without
        extracting the actual content
        """
        for qid, docid2score in self.runs.items():  # loop over the entire runfile, where some of the qids might be unwanted
            if qid not in self.qid_to_reldocs:
                continue

            for docid in docid2score:
                yield qid, docid
