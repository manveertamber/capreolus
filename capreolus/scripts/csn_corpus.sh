for lang in $1
do
  cmd="with
          rank.searcher.name=csn_distractors
          benchmark.collection.lang=$lang
          searcher.name=BM25RM3
          searcher.k1=0.9,1.0,1.1 searcher.b=0.9,1.0
          searcher.fbTerms=70,80,90,100,110 searcher.fbDocs=2,5 searcher.originalQueryWeight=0.8,0.9"

   python run.py filterrank.searcheval $cmd

done

