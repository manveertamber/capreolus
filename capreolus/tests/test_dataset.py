#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Crystina Zhang time:24/4/2020

import os
import json
import pickle
from pathlib import Path
from pprint import pprint

"""
{'nwo': 'hyj1991/easy-monitor', 'sha': '70f0a518d48157eccc8f08eb20bea3fc60fe34f0', 'path': 'src_logic/dashboard/public/dist/0.chunk.js', 
    'language': 'javascript', 'identifier': '', 'parameters': 'xac', 'argument_list': '', 'return_statement': '', 'docstring': '', 'docstring_summary': '', 'docstring_tokens': [], 'function': 'ysParseExact:!0,longDate', 'function_tokens': ['ysParseE', 'x', 'a', 'c', 't', ':!0,lo', 'g', 'D', 'a', 't', 'e'], 'url': 'https://github.com/hyj1991/easy-monitor/blob/70f0a518d48157eccc8f08eb20bea3fc60fe34f0/src_logic/dashboard/public/dist/0.chunk.js#L1-L1', 'score': 0.9434522941200334}
{'nwo': 'hyj1991/easy-monitor', 'sha': '70f0a518d48157eccc8f08eb20bea3fc60fe34f0', 'path': 'src_logic/dashboard/public/dist/0.chunk.js', 
    'language': 'javascript', 'identifier': '', 'parameters': 'plit(', 'argument_list': '', 'return_statement': '', 'docstring': '', 'docstring_summary': '', 'docstring_tokens': [], 'function': '_sub.".split("_"),weekda', 'function_tokens': ['_sub.".s', 'p', 'l', 'i', 't', '(', '"', '_', '"', ')', ',', 'wee', 'k', 'd', 'a'], 'url': 'https://github.com/hyj1991/easy-monitor/blob/70f0a518d48157eccc8f08eb20bea3fc60fe34f0/src_logic/dashboard/public/dist/0.chunk.js#L1-L1', 'score': 0.9434522941200334}
"""

# check the content of javascript and php
path = Path("/tmp/javascript_dedupe_definitions_v2.pkl")
with open(path, "rb") as f:
    a = pickle.load(f)
print("loaded")

# functions = ["ysParseExact:!0,longDate", '_sub.".split("_"),weekda']
# for i, l in enumerate(a):
#     if i % 10000 == 0:
#         print(f"{i} records processed")
#     if l["function"] in functions:
#         pprint(l)

# records the data entries with same url
url2entry = {}
for i, l in enumerate(a):
    if i % 10000 == 0:
        print(f"{i} records processed")
    url = l["url"]
    if url not in url2entry:
        url2entry[url] = [l]
    else:
        url2entry[url].append(l)

urls = list(url2entry.keys())
for url in urls:
    if len(url2entry[url]) == 1:
        del url2entry[url]


print(f"{len(url2entry)} entries has duplicated url")
json.dump(url2entry, open("java_dup_url.json", "w"))
print("finished")