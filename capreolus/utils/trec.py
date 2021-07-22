import gzip
import os
import xml.etree.ElementTree as ET
from collections import defaultdict


def threshold_trec_run(run, fold, k):
    """
    Take a trec run, and keep only the top-k docs
    """
    filtered_run = defaultdict(dict)
    # This is possible because best_search_run is an OrderedDict
    for qid, docs in run.items():
        if qid in fold["predict"]["test"]:
            for idx, (docid, score) in enumerate(docs.items()):
                if idx >= k:
                    break
                filtered_run[qid][docid] = score

    return filtered_run


def load_ntcir_topics(fn):
    topics = {}

    tree = ET.parse(fn)
    for child in tree.getroot():
        qid = child.find("qid").text.strip()
        query = child.find("content").text.strip()

        assert qid not in topics
        assert len(qid) > 0 and len(query) > 0
        topics[qid] = query

    return {"content": topics}


def load_trec_topics(queryfn):
    title, desc, narr = defaultdict(list), defaultdict(list), defaultdict(list)

    block = None
    if str(queryfn).endswith(".gz"):
        openf = gzip.open
    else:
        openf = open

    with openf(queryfn, "rt") as f:
        for line in f:
            line = line.strip()

            if line.startswith("<num>"):
                # <num> Number: 700
                qid = line.split()[-1]
                # no longer an int
                # assert qid > 0
                block = None
            elif line.startswith("<title>"):
                # <title>  query here
                title[qid].extend(line.strip().split()[1:])
                block = "title"
                # TODO does this sometimes start with Topic: ?
                assert "Topic:" not in line
            elif line.startswith("<desc>"):
                # <desc> description \n description
                desc[qid].extend(line.strip().split()[1:])
                block = "desc"
            elif line.startswith("<narr>"):
                # same format as <desc>
                narr[qid].extend(line.strip().split()[1:])
                block = "narr"
            elif line.startswith("</top>") or line.startswith("<top>"):
                block = None
            elif block == "title":
                title[qid].extend(line.strip().split())
            elif block == "desc":
                desc[qid].extend(line.strip().split())
            elif block == "narr":
                narr[qid].extend(line.strip().split())

    out = {}
    if len(title) > 0:
        out["title"] = {qid: " ".join(terms) for qid, terms in title.items()}
    if len(desc) > 0:
        out["desc"] = {qid: " ".join(terms).replace("Description: ", "") for qid, terms in desc.items()}
    if len(narr) > 0:
        out["narr"] = {qid: " ".join(terms) for qid, terms in narr.items()}

    return out


def load_qrels(qrelfile, qids=None, include_spam=True):
    labels = defaultdict(dict)
    with open(qrelfile, "rt") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue

            cols = line.split()
            qid, docid, label = cols[0], cols[2], int(cols[3])

            if qids is not None and qid not in qids:
                continue
            if label < 0 and not include_spam:
                continue

            labels[qid][docid] = label

    # remove qids with no relevant docs
    for qid in list(labels.keys()):
        if max(labels[qid].values()) <= 0:
            del labels[qid]

    labels.default_factory = None  # behave like normal dict
    return labels


def document_to_trectxt(docno, txt):
    s = f"<DOC>\n<DOCNO> {docno} </DOCNO>\n"
    s += f"<TEXT>\n{txt}\n</TEXT>\n</DOC>\n"
    return s


def topic_to_trectxt(qno, title, desc=None, narr=None):
    return (
        f"<top>\n\n"
        f"<num> Number: {qno}\n"
        f"<title> {title}\n\n"
        f"<desc> Description:\n{desc or title}\n\n"
        f"<narr> Narrative:\n{narr or title}\n\n"
        f"</top>\n\n\n"
    )


def anserini_index_to_trec_docs(index_dir, output_dir, expected_doc_count):
    from jnius import autoclass

    JFile = autoclass("java.io.File")
    JFSDirectory = autoclass("org.apache.lucene.store.FSDirectory")
    JIndexReaderUtils = autoclass("io.anserini.index.IndexReaderUtils")
    RAW = autoclass("io.anserini.index.IndexArgs").RAW

    index_reader_utils = JIndexReaderUtils()

    fsdir = JFSDirectory.open(JFile(index_dir).toPath())
    reader = autoclass("org.apache.lucene.index.DirectoryReader").open(fsdir)

    docids = set()
    for i in range(expected_doc_count):
        try:
            docid = index_reader_utils.convertLuceneDocidToDocid(reader, i)
            docids.add(docid)
        except:  # lgtm [py/catch-base-exception]
            # we reached the end?
            pass

    if len(docids) != expected_doc_count:
        raise ValueError(
            f"we expected to retrieve {expected_doc_count} documents from the index, but actually found {len(docids)}"
        )

    output_handles = [gzip.open(os.path.join(output_dir, f"{i}.gz"), "wt", encoding="utf-8") for i in range(100, 200)]

    for docidx, docid in enumerate(sorted(docids)):
        # parse documents according to here: https://github.com/castorini/anserini/blob/anserini-0.9.3/src/main/java/io/anserini/index/IndexUtils.java#L345-L352
        doc = index_reader_utils.document(reader, docid).getField(RAW)
        if doc is None:
            raise ValueError(f"{RAW} documents cannot be found in the index.")
        doc = doc.stringValue().lstrip("<TEXT>").rstrip("</TEXT>").strip()

        txt = document_to_trectxt(docid, doc)
        handleidx = docidx % len(output_handles)
        print(txt, file=output_handles[handleidx])

    for handle in output_handles:
        handle.close()


def select_passage(runs, delimiter, operation="max"):
    """for each docid, split it into (docid, segid) according to delimiter. only the largest score for each doc id are kept"""
    op2fn = {"max": max, "first": lambda lst: lst[0], "average": lambda lst: sum(lst) / len(lst)}

    operation_fn = op2fn[operation]
    for qid in runs:
        docid2scores = defaultdict(list)
        for docid, score in runs[qid].items():
            root_docid = docid.split(delimiter)[0]
            docid2scores[root_docid].append(score)

        runs[qid] = {root_docid: operation_fn(docid2scores[root_docid]) for root_docid in docid2scores}

    return runs
