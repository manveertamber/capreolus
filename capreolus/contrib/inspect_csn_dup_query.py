import os
import gzip
import json
from pathlib import Path
from collections import defaultdict

"""
This file compares how many query are actually same with each other after tokenization 
"""
import sys
if "/home/xinyu1zhang/mpi-spring/capreolus" not in sys.path:
    sys.path.append("/home/xinyu1zhang/mpi-spring/capreolus")

import numpy as np
from capreolus.utils.common import remove_newline, get_code_parser

LANGS = ["ruby", "go", "javascript", "python", "java", "php"]


# helper
def gen_doc_from_gzdir(dir):
    """ generate parsed dict-format doc from all jsonl.gz files under given directory """
    for fn in sorted(dir.glob("*.jsonl.gz")):
        f = gzip.open(fn, "rb")
        for doc in f:
            yield json.loads(doc)


# helper
def process_sentence(parser, sent):
    if isinstance(sent, list):
        sent = " ".join(sent)
    sent = remove_newline(sent)
    return parser(sent)


def check_line_no(data_dir, langs):
    def gen_file_from_gzdir(dir):
        for fn in sorted(dir.glob("*.jsonl.gz")):
            yield fn

    for lang in langs:
        lang_nlines = 0
        for set_name in ["train", "valid", "test"]:
            set_path = data_dir / lang / "final" / "jsonl" / set_name
            nlines = 0
            for fn in gen_file_from_gzdir(set_path):
                f = gzip.open(fn, "rb")
                nlines += len(f.readlines())
            print("\t", set_name, nlines)
            lang_nlines += nlines
        print(lang, lang_nlines)


def prep_parse_topic_file(data_dir, outp_dir, langs):
    camel_parser = get_code_parser(remove_punc=False)
    camel_punc_parser = get_code_parser(remove_punc=True)

    for lang in langs:
        print(f"processing langugage: {lang}")
        dirname = os.path.join(outp_dir, lang)
        os.makedirs(dirname, exist_ok=True)
        outp_camel = open(os.path.join(dirname, "withcamel"), "w", encoding="utf-8")
        outp_camel_punc = open(os.path.join(dirname, "withcamel-remove_punc"), "w", encoding="utf-8")
        outp_qid2code = open(os.path.join(dirname, "qid2code"), "w", encoding="utf-8")  # store the code for each query for later comparison

        qid = 0
        for set_name in ["train", "valid", "test"]:
            set_path = data_dir / lang / "final" / "jsonl" / set_name
            for doc in gen_doc_from_gzdir(set_path):
                url = doc["url"]
                code, ori_docstring = " ".join(doc["code_tokens"]), " ".join(doc["docstring_tokens"])
                code = process_sentence(parser=lambda x:x, sent=code)  # only remove newline for code, dont tokenize

                camel_docstring = " ".join(process_sentence(camel_parser, ori_docstring).split()[:1020])
                camel_punc_docstring = " ".join(process_sentence(camel_punc_parser, ori_docstring).split()[:1020])

                qid_str = f"{lang}-{qid}"
                outp_camel.write(f"QID: {qid_str}\nori: {ori_docstring}\nparsed: {camel_docstring}\n")
                outp_camel_punc.write(f"QID: {qid_str}\nori: {ori_docstring}\nparsed: {camel_punc_docstring}\n")
                outp_qid2code.write(f"QID: {qid_str}\ncode: {code}\nurl: {url}\n")
                qid += 1


def parse_topic_file(fn, save=False, inspect_dup=False):
    id2oridoc_fn, parseddoc2id_fn = f"{fn}.id2oridoc.json", f"{fn}.parseddoc2ids.json"
    if os.path.exists(id2oridoc_fn) and os.path.exists(parseddoc2id_fn):
        id2oridoc, parseddoc2ids = json.load(open(id2oridoc_fn)), json.load(open(parseddoc2id_fn))
    else:
        id2oridoc, parseddoc2ids = {}, defaultdict(list)
        qidst, orist, parsest = "QID:", "ori:", "parsed:"
        with open(fn, "r", encoding="utf-8") as f:
            while True:
                qid_line, ori_line, parse_line = f.readline().strip(), f.readline().strip(), f.readline().strip()
                if all([line == "" for line in [qid_line, ori_line, parse_line]]):
                    break

                assert qid_line.startswith(qidst) and ori_line.startswith(orist) and parse_line.startswith(parsest)
                qid, ori, parsed = qid_line.replace(qidst, ""), ori_line.replace(orist, ""), parse_line.replace(parsest, "")
                qid, ori, parsed = qid.strip(), ori.strip(), parsed.strip()
                id2oridoc[qid] = ori
                parseddoc2ids[parsed].append(qid)
        if save:
            json.dump(id2oridoc, open(id2oridoc_fn, "w", encoding="utf-8"))
            json.dump(parseddoc2ids, open(parseddoc2id_fn, "w", encoding="utf-8"))

    print(f"ori length: {len(id2oridoc)}; parsed length: {len(parseddoc2ids)}; (uni query: {len(set(id2oridoc.values()))})")
    if inspect_dup:
        n_dup, dup_len = 0, []
        for doc, ids in parseddoc2ids.items():
            if len(ids) == 1:
                continue

            # print("after tokenize: ", doc)
            # for id in ids:
            #     print("ori: ", id2oridoc[id])
            n_dup += 1
            dup_len.append(len(ids))
        print(f"\tnumber of dup: {n_dup}, "
              f"avg dup num: {np.mean(dup_len)}"
              f"number of empty: {len(parseddoc2ids.get('', []))}")


def load_id2code(fn):
    qidst, docst, urlst = "QID:", "code:", "url:"
    id2codeurl = {}
    with open(fn, "r", encoding="utf-8") as f:
        while True:
            qid_line, doc_line = f.readline().strip(), f.readline().strip()
            url_line = f.readline().strip()
            if qid_line == "" and doc_line == "" and url_line == "":
                break
            assert qid_line.startswith(qidst) and doc_line.startswith(docst)
            qid, code = qid_line.replace(qidst, "").strip(), doc_line.replace(docst, "").strip()
            url = url_line.replace(urlst, "").strip()
            id2codeurl[qid] = (code, url)
    return id2codeurl


def inspect_originally_same_query(id2oriquery, id2code, outp_dir):
    def get_repo_name_from_url(link):
        return "/".join(link.split("/")[:4])  # https://github.com/thooams/Ui-Bibz/xxx -> https://github.com/thooams

    query2ids = defaultdict(list)
    for id, q in id2oriquery.items():
        query2ids[q].append(id)
    # print(f"\tall judgement: {len(id2oridocstring)}, unique query: {len(query2ids)}")

    outp_fn = os.path.join(outp_dir, "docs_sharing_same_query.txt")
    url_fn = os.path.join(outp_dir, "query2url.json")
    query2ids = {q: ids for q, ids in query2ids.items() if len(ids) > 1}
    query2url = {}
    with open(outp_fn, "w", encoding="utf-8") as fout:
        for q, ids in query2ids.items():
            fout.write(f"query: {q}\n")
            for i, id in enumerate(ids):
                code, url = id2code[id]
                fout.write(f"DOC-{i}:\n\t{code}\n\t{url}\n")
            fout.write("\n\n")

            urls = list(set([get_repo_name_from_url(id2code[i][1]) for i in ids]))
            query2url[q] = urls if len(urls) == 1 else [id2code[id] for id in ids]

    print(f"\tnumber of dup query: {len(query2ids)}, "
          f"avg size of each dup: {np.mean([len(ids) for ids in query2ids.values()])}", end="\t")
    url_len = [len(u) for q, u in query2url.items()]
    assert len(url_len) == len(query2ids)
    print(f"where {url_len.count(1)} duplicate queries come from same repo")
    json.dump(query2url, open(url_fn, "w", encoding="utf-8"))


langs = LANGS

tmp_dir = Path("/tmp/csn/")
outp_dir = "./inspect_csn_query_dup"

# task 1
# prep_parse_topic_file(data_dir=tmp_dir, outp_dir=outp_dir, langs=langs)

# task 2
# langs = ["ruby"]
for lang in langs:
    outp_camel, outp_camel_punc = os.path.join(outp_dir, lang, "withcamel"), os.path.join(outp_dir, lang, "withcamel-remove_punc")
    outp_qid2code = os.path.join(outp_dir, lang, "qid2code")

    print(lang)
    for fn in [outp_camel, outp_camel_punc]:
        print(f"\t{os.path.basename(fn)}", end="\t")
        parse_topic_file(fn, save=True, inspect_dup=True)

    id2oridocstring = json.load(open(f"{outp_camel}.id2oridoc.json"))  # choose either of the doc
    id2code = load_id2code(outp_qid2code)
    inspect_originally_same_query(id2oridocstring, id2code, outp_dir=os.path.join(outp_dir, lang))


# task 3
# check_line_no(data_dir=tmp_dir, langs=langs)