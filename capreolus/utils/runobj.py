import pytrec_eval
from pathlib import Path
from collections import defaultdict, OrderedDict

from tqdm import tqdm
import numpy as np

from multiprocessing import Pool


class Runobj:
    def __init__(self, filename):
        """if the filename does not exist, a new file would be created"""
        self.filename = filename
        self.docids = []
        self.qid2pos = OrderedDict()  # (startpos, offset)

        self.fully_loaded = False
        if not self.filename.exists():  # create the file
            self.filename.parent.mkdir(exist_ok=True, parents=True)
            f = open(self.filename, "w")
            f.close()
        self.opened_file = open(self.filename)
        self.qidinfo_iterator = self.next_qidinfo()  # used in __getitem__ to find unfetched qids

    @staticmethod
    def lines2rundict(lines):
        single_run = defaultdict(dict)
        for line in lines:
            qid, _, docid, rank, score, _ = line.strip().split()
            single_run[qid][docid] = float(score)
        assert len(single_run) == 1
        return single_run

    def _get_qidlines(self, qid):
        if qid not in self.qid2pos:
            print(f"Error: {qid} not found")
            return []

        start, offset = self.qid2pos[qid]
        f = open(self.filename)
        f.seek(start, 0)
        return f.read(offset).strip().split("\n")

    def next_qidinfo(self):
        # todo: support two 'next_qidinfo'
        last_qid, last_pos = None, 0
        for qid in self.qid2pos:  # first iterate over the known ones
            lines = self._get_qidlines(qid)
            yield self.lines2rundict(lines)

        lines = []
        while True:  # then continue reading from file (if the file is finished, this part would terminate immediately)
            # problem: when the first iter is not yet finished, the second iter would read from prev point
            line = self.opened_file.readline()
            if not line:
                break

            qid, _, docid, rank, score, _ = line.strip().split()
            if docid not in self.docids:
                self.docids.append(docid)

            if qid not in self.qid2pos:
                if last_qid:
                    start = self.qid2pos[last_qid][0]
                    self.qid2pos[last_qid] = [start, last_pos-start]
                    yield self.lines2rundict(lines)
                self.qid2pos[qid] = [last_pos, -1]
                last_qid = qid
                lines = []

            last_pos = self.opened_file.tell()
            lines.append(line.strip())

        start = self.qid2pos[last_qid][0]
        self.qid2pos[last_qid] = [start, last_pos-start]
        yield self.lines2rundict(lines)  # the final qid

        self.fully_loaded = True

    def keys(self):
        for qidinfo in self.next_qidinfo():
            yield list(qidinfo.keys())[0]

    def values(self):
        if self.docids:
            for docid in self.docids:
                yield docid

        with open(self.filename) as f:
            for line in f:
                qid, _, docid, rank, score, _ = line.strip().split()
                if docid not in self.docids:
                    self.docids.append(docid)
                    yield docid

    def items(self):
        for qid in self.keys():
            yield qid, self[qid]

    def __getitem__(self, qid):
        if qid in self.qid2pos:
            lines = self._get_qidlines(qid)
            return self.lines2rundict(lines)[qid]
        else:
            for qidinfo in self.qidinfo_iterator:
                if qid in qidinfo:
                    return qidinfo[qid]
            raise KeyError(f"Unfound {qid}")

    def evaluate(self, qrels, get_evaluator_fn, metrics=[]):
        """ all qids in qrels would be evaluated """
        qrel_qids, run_qids = list(qrels.keys()), [q for q in self.keys()]
        if len(qrel_qids) != len(run_qids):  # more accurate one
            print(f"Mismatch qrels and run qids: {len(qrel_qids)} in qrels and {len(run_qids)} in run file")
        qids = sorted(list(set(qrel_qids) & set(run_qids)), key=lambda x: int(x))
        qid2scores = {}
        # with Pool(10) as p:
        for i in tqdm(range(0, len(qids), 1000)):
            cur_qids = qids[i:i+1000]
            qid_score_list = [
                get_evaluator_fn({qid: qrels[qid]}).evaluate({qid: self[qid]}) for qid in cur_qids]
            qid_score_list = [
                (qid, {m: metric2score.get(m, -1) for m in metrics})
                for qid2metric2score in qid_score_list for qid, metric2score in qid2metric2score.items()]
            qid2scores.update(dict(qid_score_list))
        return qid2scores

    def reset(self):
        self.opened_file.close()  # will need to be closed elsewhere
        self.docids = []
        self.qid2pos = OrderedDict()  # (startpos, offset)

        self.fully_loaded = False
        self.qidinfo_iterator = self.next_qidinfo()  # used in __getitem__ to find unfetched qids

    def add(self, source_path, qids):
        self.reset()
        with open(self.filename, "a") as f, open(source_path) as fin:
            for line in fin:
                qid = line.split()[0]
                if qid in qids:
                    f.write(line)
        self.opened_file = open(self.filename)

    @staticmethod
    def interpolate(runobj1, runobj2, outp_fn):
        pass


if __name__ == "__main__":
    runfile = Path(__file__).parent.parent.parent / "test" / "testrun"
    runobj = Runobj(runfile)
    # run30 = runobj["30"]
    # print(run30.keys(), len(run30["30"]))
    # print([q for q in runobj.keys()])
    # run40 = runobj["40"]
    # print(run40.keys(), len(run40["40"]))

    qrels = {
        "1": {"26559195": 1}, "2": {"29061835": 1}, "3": {"28516360": 1}, "4": {"29903896": 1}, "5": {"27388325": 1}
    }

    def get_evaluator_fn(qrel):
        return pytrec_eval.RelevanceEvaluator(qrel, {"map"}, relevance_level=1)

    scores = runobj.evaluate(qrels=qrels, get_evaluator_fn=get_evaluator_fn, metrics=["map"])
    print(scores)