#args="searcher.name=msmarcopsg_searcher benchmark.name=msmarcopsg"
args="searcher.name=BM25 searcher.k1=0.8 searcher.b=0.6 benchmark.name=msmarcopsg"
#args="searcher.name=BM25 searcher.k1=0.9 searcher.b=0.4 benchmark.name=robust04"

search=$1
eval=$2

if $search; then
  python run.py rank.search with $args
fi

if $eval; then
  python run.py rank.evaluate with $args
fi


