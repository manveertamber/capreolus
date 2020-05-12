import os
import logging
import pickle
import argparse

import wandb
from tqdm import tqdm
import elasticsearch as es

from utils import download_file


ES_AUTH_PATH = "./csn/es_auth"
LANGS = ["python", "java", "go", "php", "javascript", "ruby"]
CODE_FEATURES = ["identifier", "parameters", "argument_list", "return_statement", "function_tokens", "url"]


def _load_qids(p, qid_source):
    """
    :param p: path to file
    :param qid_source: challenge / php / python ... language name
    :return:
    """
    if not os.path.exists(p):
        if qid_source != "challenge":
            raise FileNotFoundError(f"provided path {p} cannot be resolved")

        url = "https://raw.githubusercontent.com/github/CodeSearchNet/master/resources/queries.csv"
        download_file(url, p)
        assert os.path.exists(p)

    with open(p) as f:
        queries = []
        for i, line in enumerate(f):
            if i == 0:  # the first line is "query"
                continue
            queries.append(line.strip())
    return queries


def _load_esauth(p):
    with open(p) as f:
        cloud_id = f.readline().strip()
        name, psw = f.readline().strip().split()
    return cloud_id, (name, psw)


def load_docs(path, lang):
    langf = open(os.path.join(path, f"{lang}_dedupe_definitions_v2.pkl"), "rb")
    dedup_codes = pickle.load(langf)
    docs = [
        {feature: (" ".join(line[feature]) if feature == "function_tokens" else line[feature])
            for feature in CODE_FEATURES}
        for line in dedup_codes
    ]
    return docs


def init_esobj():
    cloud_id, (name, psw) = _load_esauth(ES_AUTH_PATH)
    return es.Elasticsearch(cloud_id=cloud_id, http_auth=(name, psw))


def prep_index(es_obj, doc_path, langs, parallel=True, chunk_size=3000, overwrite=False):
    """
    Args:
        es_obj: elasticsearch.ElasticSearch object
        doc_path: str, root path to the {lang}.dedup.pkl files
        langs: list, language to build index on
        parallel: bool, whether to use bulk. use index() if False. default to be True
        chunk_size: int, number of documents in one bulk, only be used if parallel=True. default to be 5000
        overwrite: bool, whether overwrite previouly build index
    Returns:
        elasticsearch.ElasticSearch object with index built
    """
    for lang in langs:
        index_name = f"code-search-{lang}"
        if es_obj.indices.exists(index=index_name) and not overwrite:
            print(f"index {index_name} already exist, skip indexing process")
            continue

        if es_obj.indices.exists(index=index_name) and overwrite:
            print(f"deleting existing index")
            es_obj.indices.delete(index=index_name)

        es_obj.indices.create(index=index_name, ignore=400)  # 400: index already exists
        print(f"preparing {index_name}...")
        docs = load_docs(doc_path, lang)
        if not parallel:
            for doc in tqdm(docs):  # doc: a dict with CODE_FEATURES as key
                es_obj.index(index=index_name, body=doc)  # TODO: id=f"{lang}-{i}"
        else:
            all_docs = []
            for doc in docs:
                all_docs.extend([
                    {"index": {"_index": index_name, "_type": "document"}},
                    doc,
                ])
            for i in tqdm(range(0, len(all_docs), chunk_size)):
                es_obj.bulk(all_docs[i:i+chunk_size], request_timeout=30)  # default timeout=10 can easily cause error

    return es_obj


def search(es_obj, queries, langs, outp_fn, nhits, output_rerun_file, output_predict_csv):
    if (not output_rerun_file) and (not output_predict_csv):
        raise ValueError(f"one of rerank file or pred file should be prepared")

    filename, ext = os.path.splitext(outp_fn)
    filedir = os.path.dirname(outp_fn)
    if output_predict_csv:
        pred_path = os.path.join(filedir, "model_predictions.csv")
        pred_file = open(pred_path, "w", encoding="utf-8")
        pred_file.write("query,language,identifier,url\n")

    for lang in langs:
        print(f"searching {lang}")
        if output_rerun_file:
            outp_content_f = open(os.path.join(filedir, 'esearch.content'), "w", encoding="utf-8")
            outp_index_f = open(os.path.join(filedir, 'esearch.index'), "w", encoding="utf-8")

        for q in tqdm(queries):
            results = es_obj.search(
                index=f"code-search-{lang}",
                body={"query": {"bool": {"must": {"query_string": {"query": q, }}}}},
                size=nhits,
            )
            docs = results["hits"]["hits"]
            for doc in docs:
                score, src = doc["_score"], doc["_source"]
                if output_rerun_file:
                    try:
                        function = src['function_tokens'].replace('\n', ' ').replace('\r', ' ')
                        url = src['url']
                        outp_content_f.write(f"{q}\n{function}\n")
                        outp_index_f.write(f"{q}\t{lang}\t{url}\n")
                    except KeyError:
                        print("Key Error Detected", src.keys())
                if output_predict_csv:
                    id, url = src["identifier"], src["url"]
                    id = "-".join(id.split(",")) if len(id) > 2 else ""  # in case , is included in identifier
                    pred_file.write(f"{q},{lang},{id},{url}\n")
        if output_rerun_file:
            outp_content_f.close()
            outp_index_f.close()

    if output_predict_csv:
        pred_file.close()
        wandb.init(project="code-search")
        wandb.save(pred_path)


def prep_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_root", "-p", required=True, type=str)
    parser.add_argument("--output_fn", "-outp", required=True, type=str)
    parser.add_argument("--nhits", "-n", default=10, type=int)
    parser.add_argument("--langs", "-l", default="all", choices=LANGS+["all"], type=str)
    parser.add_argument("--overwrite", "-w", action='store_true')
    parser.add_argument("--output_rerun_file", action='store_true')
    parser.add_argument("--output_predict_csv", action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = prep_args()
    langs = LANGS if args.langs == "all" else args.langs
    if isinstance(langs, str):
        langs = [langs]

    es_obj = init_esobj()
    prep_index(es_obj, doc_path=args.path_root, langs=langs, overwrite=args.overwrite)

    queries = _load_qids("../CSNdata/queries.csv")
    search(es_obj, queries, langs,
           outp_fn=args.output_fn,
           nhits=args.nhits,
           output_rerun_file=args.output_rerun_file,
           output_predict_csv=args.output_predict_csv)
