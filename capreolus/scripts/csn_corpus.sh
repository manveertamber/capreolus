cmd="filterrank.search with
        rank.searcher.name=csn_distractors
        benchmark.collection.lang=$1
        searcher.name=BM25_reranker"
echo $cmd
python run.py $cmd
