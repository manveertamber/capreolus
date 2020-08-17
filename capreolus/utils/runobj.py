import tarfile

from capreolus.utils.common import OrderedDefaultDict
from capreolus.utils.loginit import get_logger


logger = get_logger(__name__)


class Runs:
    supported_format = ["trec", "tar"]

    def __init__(self, runfile, format="trec", buffer=True):
        """ TODO: support init from runs """
        if format not in self.supported_format:
            raise ValueError(f"Unrecognized format: {format}, expected to be one of {' '.join(self.supported_format)}")

        self.runfile = runfile
        self.format = format
        self.buffer = buffer
        self.qid2position = {}

        self.bytef = open(self.runfile, "rb")
        self.prepare_qid2pos()


    def __del__(self):
        self.bytef.close()

    @staticmethod
    def load_trec_run(fn):
        # Docids in the run file appear according to decreasing score, hence it makes sense to preserve this order
        run = OrderedDefaultDict()

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
            qids = sorted(preds.keys(), key=lambda k: int(k))
            for qid in qids:
                rank = 1
                for docid, score in sorted(preds[qid].items(), key=lambda x: x[1], reverse=True):
                    print(f"{qid} Q0 {docid} {rank} {score} capreolus", file=outf)
                    rank += 1
                    count += 1

    @staticmethod
    def from_runfile(format="trec", buffer=True):
        """
        :param format: str, input runfile format, must be one of `self.supported_format`
        :return: a Runs object
        """
        if format == "trec" and not buffer:
            pass

    def _open_runfile(self):
        return open(self.runfile) if self.format == "trec" else tarfile.open(self.runfile)

    def prepare_qid2pos(self):
        last_end = 0  # the cursor position after last read()
        assert not self.qid2position
        f = self._open_runfile()
        while True:
            line = f.readline()
            if not line:
                break

            qid = line.split()[0]
            if qid in self.qid2position:
                start = self.qid2position[qid][0]
                self.qid2position[qid][1] = f.tell() - start
            else:  # new qid
                self.qid2position[qid] = [last_end, f.tell()]

            last_end = f.tell()

    def keys(self):
        for qid in self.qid2position:
            yield qid

    def docids(self):
        assert self.qid2position
        docids = set()
        # don't maintain this in obj attribute since
        # (1) it might change after some qid is deleted (2) reduce the deepcopy overhead
        f = self._open_runfile()
        for line in f:
            line = line.split()
            qid, docid = line[0], line[2]  # qid, Q0, docid, ...
            if qid not in self.qid2position:
                continue

            if docid in docids:
                continue

            docids.add(docid)
            yield docid

    def items(self):
        assert self.qid2position  # must be called after qid2position is fully prepared

        # assume records with same qid is same
        f = self._open_runfile()
        last_qid = -1
        doc2score = {}
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            if qid not in self.qid2position:  # ignore deleted terms
                continue

            if last_qid == -1:
                last_qid = qid

            if qid != last_qid:
                yield last_qid, doc2score

                last_qid = qid
                doc2score = {}

            doc2score[docid] = float(score)

        f.close()
        yield qid, doc2score  # the last item

    def __iter__(self):
        return iter(self.keys())

    def __getitem__(self, qid):
        if qid not in self.qid2position:
            raise KeyError(f"Cannot find {qid} in current run")
        start, offset = self.qid2position[qid]
        self.bytef.seek(start, 0)  # set cursor to position `start`, counting from the beginning of the file
        lines = self.bytef.read(offset).decode("utf-8").rstrip("\n").split("\n")
        docids = [l.split()[2] for l in lines]
        return docids

    def __delitem__(self, qid):
        if qid not in self.qid2position:
            raise KeyError(f"Cannot find {qid} in current run")
        del self.qid2position[qid] # TODO: delete corresponding doc terms

    def evaluate(self, eval_runs_fn, qids=None):
        """
        :param eval_runs_fn: a function take `trec runs` as parameter and return {qid: {metric: score}}
        :param qids: an iterator, containing qids expected to return. Optional
            all qids of current run are returned if None
        :return: evaluated runs
        """
        if self.buffer:
            scores = {qid: eval_runs_fn({qid: doc2score}).get(qid, {}) for qid, doc2score in self.items() if not qids or qid in qids}
            scores = {qid: score for qid, score in scores.items() if score}  # filter unevaluated qids
        else:
            runs = self.load_trec_run(self.runfile)
            if qids:
                runs = {qid: runs[qid] for qid in runs if qid in qids}
            scores = eval_runs_fn(runs)
        return scores
