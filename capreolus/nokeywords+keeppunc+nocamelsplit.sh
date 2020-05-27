do_train=$1
do_eval=$2
out_file="csn.ranking.scores"

echo "running BM25"
for lang in "ruby" "python" # "go" "javascript" "java" "php" "python" 
# for lang in "python" "php"  # "ruby" "go" "javascript" "java"  # "ruby" "go" "python" 
do
	echo "processing language $lang"
	args="collection=codesearchnet benchmark=codesearchnet_corpus 
		searcher=BM25_reranker 
		searcher.k1=1.3~1.2~1.1~1.0~0.9 
		searcher.b=0.9~1.0  
		searcher.hits=1000 
		searcher.searcher.includetrain=False  
		searcher.searcher.benchmark.remove_keywords=False collection.remove_keywords=False benchmark.remove_keywords=False
		searcher.searcher.benchmark.lang=$lang collection.lang=$lang benchmark.lang=$lang"


	# searcher.searcher.benchmark.camelstemmer=False collection.camelstemmer=False benchmark.camelstemmer=False
	# searcher.searcher.benchmark.remove_punc=False collection.remove_punc=False benchmark.remove_punc=False
	# searcher.searcher.benchmark.remove_punc=False collection.remove_punc=False benchmark.remove_punc=False
	echo $args

	if $do_train 
	then
		echo "training"
		# python run.py rank.train with $args 
		python run.py nonnn_rerank.train with $args 
	fi


	if $do_eval 
	then
		echo "evaluating"
		# python run.py rank.evaluate with $args 
		python run.py nonnn_rerank.evaluate with $args 
	fi
done

