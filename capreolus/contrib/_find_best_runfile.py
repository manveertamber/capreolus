#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Crystina Zhang time:25/5/2020
import os
import glob
import sys
from tqdm import tqdm
from argparse import ArgumentParser

if "/home/xinyu1zhang/mpi-spring/capreolus" not in sys.path:
    sys.path.append("/home/xinyu1zhang/mpi-spring/capreolus")

# from capreolus.benchmark import CodeSearchNetCorpus
from capreolus.evaluator import eval_runfile

parser = ArgumentParser()
parser.add_argument("--rundir", "-r", type=str, required=True)
args = parser.parse_args()

runfile_dir = args.rundir


# lang = "python"
# camelstemmer = True
# remove_keywords = False
# cfg = {"_name": "codesearchnet_corpus", "lang": lang, "camelstemmer": camelstemmer}

best_score, best_file = -1, None
existing_best_file = glob.glob(f"{runfile_dir}/*best*")

qrels, folds = "", ""


if existing_best_file:
    print(f"best file found: {existing_best_file}")
else:
    for runfile in tqdm(glob.glob(f"{runfile_dir}/*")):
        score = eval_runfile(runfile, qrels, ["mrr"])  # use the dev + test as an approximation
        if score["mrr"] > best_score:
            best_score = score["mrr"]
            best_file = runfile
            print(f"new best score: {best_score} - {os.path.basename(best_file)}")

    base_best_file = os.path.basename(best_file)
    best_fn_copy = os.path.join(runfile_dir, f"s1.best.{base_best_file}")
    with open(best_file, "r") as fin, open(best_fn_copy, "w") as fout:
        for line in tqdm(fin.readlines(), desc="copying files"):
            fout.write(line)