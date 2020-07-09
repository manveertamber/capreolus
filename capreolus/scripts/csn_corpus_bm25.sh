for lang in $1
do
  cmd="with
          rank.searcher.name=csn_distractors
          benchmark.collection.lang=$lang
          searcher.name=BM25_reranker
          searcher.k1=0.9,1.0,1.1 searcher.b=0.9,1.0"

python run.py filterrank.searcheval $cmd
done
