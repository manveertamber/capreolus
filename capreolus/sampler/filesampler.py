import hashlib
from pathlib import Path

import torch
import numpy as np
from capreolus import ConfigOption, Dependency
from capreolus.utils.common import get_logger
from capreolus.sampler import Sampler, TrainingSamplerMixin
from capreolus.utils.exceptions import MissingDocError


logger = get_logger(__name__)


@Sampler.register
class TrainTripletFileSampler(Sampler, TrainingSamplerMixin, torch.utils.data.IterableDataset):
    module_name = "tripletFile"
    config_spec = [
        ConfigOption("path", None, "Triple file path, each line should contain qid\tpos-docid\tneg-docid"),
    ]

    def prepare(self, qid_to_docids, qrels, extractor, relevance_level=1, **kwargs):
        self.extractor = extractor
        self.build()

    def clean(self):  # oveerwrite the clean function in TrainingSamplerMixin
        logger.info("No cleaning is performed under msmarco sampler")

    def get_hash(self):
        qids = {line.strip().split()[0] for line in open(self.triplet_file)}
        sorted_rep = sorted(list(qids))
        key_content = "{0}{1}{2}".format(self.extractor.get_cache_path(), str(sorted_rep), self.config["path"])
        key = hashlib.md5(key_content.encode("utf-8")).hexdigest()
        return "triplet_{0}".format(key)

    def build(self):
        self.triplet_file = self.config["path"]

    def __iter__(self):
        # when running in a worker, the sampler will be recreated several times with same seed,
        # so we combine worker_info's seed (which varies across obj creations) with our original seed
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # avoid reseeding the same way multiple times in case DataLoader's behavior changes
            if not hasattr(self, "_last_worker_seed"):
                self._last_worker_seed = None

            if self._last_worker_seed is None or self._last_worker_seed != worker_info.seed:
                seeds = [self.config["seed"], worker_info.seed]
                self.rng = np.random.Generator(np.random.PCG64(seeds))
                self._last_worker_seed = worker_info.seed

        return iter(self.generate_samples())

    def generate_samples(self):
        with open(self.triplet_file) as f:
            # todo: make it able to randomly select from the whole document rather than reading sequentially
            for line in f:
                qid, posdocid, negdocid = line.strip().split()
                try:
                    yield self.extractor.id2vec(qid, posdocid, negdocid, label=[1, 0])
                except MissingDocError:
                    logger.warning(
                        "skipping training pair with missing features: qid=%s posid=%s negid=%s", qid, posdocid, negdocid
                    )
