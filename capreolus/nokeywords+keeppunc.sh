do_train=$1
do_eval=$2
out_file="csn.ranking.scores"

echo "running BM25"
# for lang in "go" "javascript" "java" "python" "php" # "ruby" "go" "python" 
# for lang in "ruby"
for lang in "ruby"
do
	echo "processing language $lang"
	args="collection=codesearchnet benchmark=codesearchnet_corpus 
		searcher=BM25_reranker 
		searcher.k1=1.3~1.2~1.1~1.0~0.9~0.8 
		searcher.b=0.7~0.8~0.9~1.0  
		searcher.hits=1000 
		searcher.searcher.includetrain=False  
		searcher.searcher.benchmark.lang=$lang collection.lang=$lang benchmark.lang=$lang"

	# searcher.searcher.benchmark.remove_punc=False collection.remove_punc=False benchmark.remove_punc=False
	echo $args

	if $do_train 
	then
		echo "training"
		python run.py nonnn_rerank.train with $args 
	fi


	if $do_eval 
	then
		echo "evaluating"
		python run.py nonnn_rerank.evaluate with $args 
	fi
done

