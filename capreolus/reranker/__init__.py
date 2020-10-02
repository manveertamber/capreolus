import os
import pickle
import torch

from capreolus import Dependency, ModuleBase
from capreolus.utils.loginit import get_logger

logger = get_logger(__name__)

class Reranker(ModuleBase):
    """Base class for Reranker modules. The purpose of a Reranker is to predict relevance scores for input documents. Rerankers are generally supervised methods implemented in PyTorch or TensorFlow.

    Modules should provide:
        - a ``build_model`` method that initializes the model used
        - a ``score`` and a ``test`` method that take a representation created by an :class:`~capreolus.extractor.Extractor` module as input and return document scores
        - a ``load_weights`` and a ``save_weights`` method, if the base class' PyTorch methods cannot be used
    """

    module_type = "reranker"
    dependencies = [
        Dependency(key="extractor", module="extractor", name="embedtext"),
        Dependency(key="trainer", module="trainer", name="pytorch"),
    ]

    def add_summary(self, summary_writer, niter):
        """
        Write to the summay_writer custom visualizations/data specific to this reranker
        """
        for name, weight in self.model.named_parameters():
            summary_writer.add_histogram(name, weight.data.cpu(), niter)
            # summary_writer.add_histogram(f'{name}.grad', weight.grad, niter)

    def save_weights(self, weights_fn, optimizer):
        if not os.path.exists(os.path.dirname(weights_fn)):
            os.makedirs(os.path.dirname(weights_fn))

        d = {k: v for k, v in self.model.state_dict().items() if ("embedding.weight" not in k and "_nosave_" not in k)}
        with open(weights_fn, "wb") as outf:
            pickle.dump(d, outf, protocol=-1)

        optimizer_fn = weights_fn.as_posix() + ".optimizer"
        with open(optimizer_fn, "wb") as outf:
            pickle.dump(optimizer.state_dict(), outf, protocol=-1)

    def load_weights(self, weights_fn, optimizer=None):
        with open(weights_fn, "rb") as f:
            # d = pickle.load(f)
            d = torch.load(f, map_location=self.trainer.device)

        cur_keys = set(k for k in self.model.state_dict().keys() if not ("embedding.weight" in k or "_nosave_" in k))
        missing = cur_keys - set(d.keys())
        overlap_keys = cur_keys & set(d.keys())
        if len(missing) > 0:
            # raise RuntimeError("loading state_dict with keys that do not match current model: %s" % missing)
            logger.warning("loading state_dict with keys that do not match current model: %s" % missing)
            for k in missing:
                print(k, self.model.state_dict()[k].size())

            d = {k: d[k] for k in d if k not in missing}
        if len(overlap_keys) < len(d):
            d = {k: d[k] for k in d if k in overlap_keys}
        d["combine.weight"] = torch.ones([1, 3]).to("cuda:0")
        print("number of overlap_Keys", len(overlap_keys))

        for k in d:
            if k not in overlap_keys:
                print(k)
        print("not overlap but in d: ", len(set(d.keys()) - overlap_keys))
        old = {k: torch.tensor(v) for k, v in self.model.state_dict().items()}

        # self.model.load_state_dict(d, strict=False)
        self.model.load_state_dict(d, strict=True)
        '''
        for k, v in self.model.state_dict().items():
            print(k, f"old: ", (v - old[k]).abs().sum(), "new: ", (v - d[k]).abs().sum())
        '''

        if optimizer:
            optimizer_fn = weights_fn.as_posix() + ".optimizer"
            if os.path.exists(optimizer_fn):
                with open(optimizer_fn, "rb") as f:
                    optimizer.load_state_dict(pickle.load(f))


from profane import import_all_modules


import_all_modules(__file__, __package__)
