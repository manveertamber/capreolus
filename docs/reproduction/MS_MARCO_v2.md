# Capreolus: MaxP Reranking Baselines on MS MARCO v2 Retrieval

This page contains instructions for running MaxP baselines on MS MARCO v2 document and passage ranking task using Capreolus.
After [setting up](./PARADE.md#Setup) capreolus and [testing the installation](./PARADE.md#testing-installation), we are ready to start the replication.
Note that all the following scripts requires that we are on the top-level `capreolus` folder.


## Dataset preparation
While `capreoulus` would automatically download the collection, 
if you already have the dataset on the machine or you would prefer the data to be stored in a specific location,
feel free to soft link them to `capreoulus` directory as below: 
Note that the the document and passage collection are provided as .gz format,
but `capreolus` would require them in plain text format (uncompressed).

If this is not the case, feel free to skip this section and jump directly into [Training](#Training) and [Inference](#Inference).

### Document Retrieval
<!-- After download the topic and query files from [here](https://microsoft.github.io/msmarco/TREC-Deep-Learning.html#document-ranking-dataset) into `capreolus/data/msdoc_v2`
``` -->
```bash
mkdir -p capreolus/data/msdoc_v2
ln -s /path/to/msmarco_v2_doc capreolus/data/msdoc_v2/msmarco_v2_doc
for dir in capreolus/data/msdoc_v2/msmarco_v2_doc/*; do unzip $dir; done  # todo: 
# cat docv2_dev_qrels.tsv  docv2_train_qrels.tsv > qrels.txt
# cat 2021_queries.tsv docv2_dev_queries.tsv  docv2_train_queries.tsv > topics.txt
```
<!-- Now the `capreolus/data/msdoc_v2` should contain the following files:
```
- msmarco_v2_doc/ # which contains 60 plaintext files
- docv2_dev_qrels.tsv
- docv2_dev_queries.tsv
- docv2_train_qrels.tsv
- docv2_train_queries.tsv
- docv2_train_top100.txt
- 2021_queries.tsv -->
<!-- ``` -->
<!-- Note: if on `ceres`, the data is available under `/GW/carpet/nobackup/czhang/msdoc_v2`:
```
ln -s /GW/carpet/nobackup/czhang/msdoc_v2 capreolus/data
``` -->

### Passage Retrieval
```bash
pass_dir=capreolus/data/mspass_v2
mkdir -p $pass_dir
wget https://msmarco.blob.core.windows.net/msmarcoranking/docv2_dev_queries.tsv -P $pass_dir
wget https://msmarco.blob.core.windows.net/msmarcoranking/docv2_dev_queries.tsv -P $pass_dir
for dir in $pass_dir/msmarco_v2_passage/*; do unzip $dir; done  # todo: 

# cat passv2_dev_qrels.tsv  passv2_train_qrels.tsv > qrels.txt
# cat 2021_queries.tsv passv2_dev_queries.tsv  passv2_train_queries.tsv > topics.txt
```
<!-- 
Make sure the `capreolus/data/mspass_v2` contain the following files, which could be 
```
- msmarco_v2_passage/ # which contains 70 plaintext files
- passv2_dev_qrels.tsv
- passv2_dev_queries.tsv
- passv2_train_qrels.tsv
- passv2_train_queries.tsv
- passv2_train_top100.txt
- 2021_queries.tsv
```
 -->

<!-- Note: if on `ceres`, the data is available under `/GW/carpet/nobackup/czhang/mspass_v2`:
```
ln -s /GW/carpet/nobackup/czhang/mspass_v2 capreolus/data
``` -->

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
