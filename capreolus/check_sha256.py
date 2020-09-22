import json
import cbor
import numpy as np
from time import time

f = open("/home/xinyu1zhang/.capreolus/cache/collection-treccar/documents/paragraphcorpus/paragraphcorpus.cbor", "rb")


def print_dict(hie):
    for fir_key in hie:
        print("FIRST: ", fir_key, len(hie[fir_key]))
        for sec_key in hie[fir_key]:
            print("SECOND: ", sec_key, len(hie[fir_key][sec_key]))


i, t, hierachy = 0, time(), {}
while True:
    try:
        if i and i % 30000 == 0:
            len_info = [len(vv) for k, v in hierachy.items() for kk, vv in v.items()]
            print(i, time() - t, "avg: ", np.mean(len_info), "max: ", max(len_info), "min", min(len_info))

        tag, pid, psgs = cbor.load(f)
        code = int(pid, 16)
        fir, sec = str(code % (10**2)), str(code % (10**4))

        if fir not in hierachy:
            hierachy[fir] = {}

        if sec not in hierachy[fir]:
            hierachy[fir][sec] = [pid]
        else:
            hierachy[fir][sec].append(pid)

        i += 1
    except EOFError:
        break

json.dump(hierachy, open("tmp.hierachy.json", "wb"))
print_dict(hierachy)
