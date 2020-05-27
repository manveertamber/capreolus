import os
import sys
import math
import json
from pathlib import Path
from argparse import ArgumentParser

import numpy as np

if "/home/xinyu1zhang/mpi-spring/capreolus" not in sys.path:
    sys.path.append("/home/xinyu1zhang/mpi-spring/capreolus")
from capreolus.utils.common import load_keywords

LANGS = ["python", "java", "go", "php", "javascript", "ruby"]
KEYWORDS = {lang: load_keywords(lang) for lang in LANGS}

from jnius import autoclass
stemmer = "porter"
Analyzer = autoclass("io.anserini.analysis.DefaultEnglishAnalyzer")
analyzer = Analyzer.newStemmingInstance(stemmer)
anserini_tokenize = autoclass("io.anserini.analysis.AnalyzerUtils").analyze

parser = ArgumentParser()
parser.add_argument("--dir", "-r", type=str, required=True)
args = parser.parse_args()

runfile_dir = args.dir

configs = os.listdir(runfile_dir)

lang2key2idf = {lang: {} for lang in LANGS}
lang2key2querydf = {lang: {} for lang in LANGS}

for config in configs:
    for lang in LANGS:
        print(lang)
        idfs = []

        dir = Path(os.path.join(runfile_dir, config, lang))
        p = dir / "keywords2df.json"
        queryp = dir / "keywords2querydf.json"

        keyword2df = json.load(open(p))
        keyword2querydf = json.load(open(queryp))

        N = keyword2df["total_data_num"]
        for w, df in keyword2df.items():
            if w == "total_data_num":
                continue

            idf = (N - df + 0.5) / (df + 0.5)
            idf = math.log(1 + idf)
            if w in lang2key2idf[lang]:
                lang2key2idf[lang][w].append(idf)
            else:
                lang2key2idf[lang][w] = [idf]
            idfs.append(idf)
            # print(f"\t{w} / {df} / {idf}")

        for w, qdf_list in keyword2querydf.items():
            avg_qdf = np.mean(qdf_list)
            num_qdf = len(qdf_list)

            if w in lang2key2querydf[lang]:
                lang2key2querydf[lang][w].append((num_qdf, avg_qdf))
            else:
                lang2key2querydf[lang][w] = [(num_qdf, avg_qdf)]
        print(f"\tavg: {np.mean(idfs)}")


print("finished! start writing idf")
fout = open("new.record.idf", "w")
fout.write("lang\tkeyword\t")
for config in configs:
    fout.write(f"{config}\t")
fout.write("\n")

for lang, keyword2idfs in lang2key2idf.items():
    for w, idfs in keyword2idfs.items():
        idfs = "\t".join(map(str, idfs))
        fout.write(f"{lang}\t{w}\t{idfs}\n")
    fout.write("\n\n")
fout.close()

print("finished! start writing query idf")
fout = open("new.record.query_df", "w")
fout.write("lang\tkeyword\t")
for config in configs:
    fout.write(f"{config}\t")
fout.write("n_df / avg_df")
fout.write("\n")

for lang, key2querydf in lang2key2querydf.items():
    n_stopwords = 0
    for w, num_qdf_avg_qdf in key2querydf.items():
        dfs = "\t".join(map(lambda n_a: f"{n_a[0]}/{n_a[1]}", num_qdf_avg_qdf))
        if not anserini_tokenize(w).toArray():
            fout.write(f"{lang}\t{w}[IN STOPWORDS]\t{dfs}\n")
            n_stopwords += 1
        else:
            fout.write(f"{lang}\t{w}\t{dfs}\n")
    print(f"number of stopwords in {lang}: {n_stopwords}/{len(key2querydf)}, %.4f" % (n_stopwords/len(key2querydf)))
    fout.write("\n\n")
fout.close()

print("selective comparison: (e.g. used the remove_punc-split_camel-keep_keywords-no_anserini_tok")
fout = open("new.analysis.split_camel-keep_keywords-no_anserini_tok", "w")
config = "remove_punc-split_camel-keep_keywords-no_anserini_tok"
config_i = configs.index(config)
for lang in lang2key2idf:
    keyword2idfs, keyword2dfs = lang2key2idf[lang], lang2key2querydf[lang]
    # keyword2idf, keyword2df = keyword2idfs[config_i], keyword2dfs[keyword2idfs]
    for kw, idfs in keyword2idfs.items():
        idf = idfs[config_i]
        num_qdf, avg_qdf = keyword2dfs[kw][config_i]
        is_stop = "[STOPWORDS]" if (not anserini_tokenize(kw).toArray()) else ""
        fout.write(f"{lang}\t{kw+is_stop}\t{idf}\t{num_qdf}\t{avg_qdf}\n")

fout.close()
print("finished")

