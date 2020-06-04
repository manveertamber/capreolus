#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Crystina Zhang time:31/5/2020

import os
import tarfile
from argparse import ArgumentParser

from tqdm import tqdm
from bs4 import BeautifulSoup

import sys

sys.path.append("/home/xinyu1zhang/cikm/capreolus-covid")
from capreolus.utils.common import download_file

urls = "http://ceb.nlm.nih.gov/~robertske/pmc-0%d.tar.gz"  # 0, 1, 2, 3


def load_docs(docidfn):
    with open(docidfn) as f:
        docs = [l.strip() for l in f.readlines()]
    return docs


def get_id_text(fn):
    with open(fn, "r") as f:
        t = f.read()

    soup = BeautifulSoup(t, "lxml")
    pmcid = os.path.basename(fn).replace(".nxml", "")
    paras = soup.find("body").find_all("p")
    doc = " ".join([p.text.strip().replace("\n", " ").replace("\t", " ").replace("\r", " ") for p in paras])
    doc = " ".join(doc.split())  # remove wide space
    return pmcid, doc


def filtered_transform(cache_dir, kept_docs, outp_dir):
    exists = os.path.exists
    os.makedirs(outp_dir, exist_ok=True)
    for i in range(4):
        cache_fold = os.path.join(cache_dir, f"pmc-0{i}")
        cache_fold_tar = f"{cache_fold}.tar.gz"
        if not exists(cache_fold) and not exists(cache_fold_tar):
            url = urls % i
            print(f"downloading from url: {url}")
            download_file(url, cache_fold_tar)
        if not exists(cache_fold) and exists(cache_fold_tar):
            with tarfile.open(cache_fold_tar) as tar:
                tar.extractall(cache_dir)

        assert exists(cache_fold)
        subfolds = os.listdir(cache_fold)
        for subfold in tqdm(subfolds, desc="processing subfolds"):
            subfold = os.path.join(cache_fold, subfold)
            for pmcid_fn in os.listdir(subfold):
                pmcid = pmcid_fn.replace(".nxml", "")
                if pmcid not in kept_docs:
                    continue
                id, txt = get_id_text(os.path.join(subfold, pmcid_fn))
                with open(os.path.join(outp_dir, f"{id}.txt"), "w") as f:
                    f.write(txt)


parser = ArgumentParser()
parser.add_argument("--docids_fn", "-d", type=str, default="/home/xinyu1zhang/cikm/capreolus-covid/capreolus/contrib/cds.doc.txt")
parser.add_argument("--outp_dir", "-o", type=str, default="/home/xinyu1zhang/cikm/capreolus-covid/capreolus/cds")

args = parser.parse_args()

docids_fn = args.docids_fn
outp_dir = args.outp_dir

docs = load_docs(docids_fn)
filtered_transform("/tmp", docs, outp_dir)
