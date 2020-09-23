from pathlib import Path
from collections import defaultdict


class Runobj:
    def __init__(self, filename):
        self.filename = filename
        self.docids = []
        self.qid2pos = {}  # (startpos, offset)

        self.fully_loaded = False
        self.opened_file = open(filename)
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

    def __getitem__(self, qid):
        if qid in self.qid2pos:
            lines = self._get_qidlines(qid)
            return self.lines2rundict(lines)
        else:
            for qidinfo in self.qidinfo_iterator:
                if qid in qidinfo:
                    return qidinfo
            raise KeyError(f"Unfound {qid}")

    def evaluate(self, evaluator):
        if self.fully_loaded:
            all_qids = self.keys()


if __name__ == "__main__":
    runfile = Path(__file__).parent.parent.parent / "test" / "testrun"
    runobj = Runobj(runfile)
    run30 = runobj["30"]
    print(run30.keys(), len(run30["30"]))
    print([q for q in runobj.keys()])
    run40 = runobj["40"]
    print(run40.keys(), len(run40["40"]))
