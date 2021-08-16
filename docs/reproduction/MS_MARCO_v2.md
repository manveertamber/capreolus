# Capreolus: MaxP Reranking Baselines on MS MARCO v2 Retrieval

This page contains instructions for running MaxP baselines on MS MARCO v2 document and passage ranking task using Capreolus.
After [setting up](./PARADE.md#Setup) capreolus and [testing the installation](./PARADE.md#testing-installation), we are ready to start the replication.
Note that all the following scripts requires that we are on the top-level `capreolus` folder.


## Training 

- To run MaxP for Document Retrieval:
```
python -m capreolus.run rerank.train with file=docs/reproduction/MS_MARCO_v2/config_msmarco_v2_doc.txt
```
- To run monoBERT for Passage Retrieval:
```
python -m capreolus.run rerank.train with file=docs/reproduction/MS_MARCO_v2/config_msmarco_v2_passage.txt
```

<!-- Expected score:
```
MAP      MRR     R@100
0.1503   0.152   0.3397
``` -->

## Inference
The above script would run inference on developement and test set by reranking *BM25*.
- If you would like to use the same checkpoint but to rerank other run files (e.g. from dense retrieval):
```python
from capreolus import parse_config_string
from capreolus.task import Rerank

external_run_path = "/path/to/external_run_file.txt"
output_path = "output_directory"

config_string="file=docs/reproduction/config_msmarco_v2_doc.txt"
task = Rerank(parse_config_string(config_string))
task.predict_external(
    external_run_path=external_run_path,
    output_path=output_path,
    set_name="both",            # `both`, `dev`, `test`. `both` would rerank on both dev and test set
)
```
where the `config_string` shuold be exactly the same one passed to the `rerank.train` command above (the lines after `with`).

- Or if you would like to use another `capreolus` checkpoint to evaluate the prepared BM25 runfile, which should be compatible with the configured `reranker`:
```python
from capreolus import parse_config_string
from capreolus.task import Rerank

external_checkpoint_path = "/path/to/checkpoint"  # the tf checkpoints should be named as `dev.best....`
output_path = "output_directory"

config_string="file=docs/reproduction/config_msmarco_v2_passage.txt"
task = Rerank(parse_config_string(config_string))
task.predict_external(
    external_checkpoint=external_checkpoint_path,
    output_path=output_path, 
    set_name="both",            # `both`, `dev`, `test`. `both` would rerank on both dev and test set
)
```
