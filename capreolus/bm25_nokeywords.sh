do_train=$1
do_eval=$2
out_file="csn.ranking.scores"

echo "running BM25"
# for lang in "go" "javascript" "java" "python" "php" # "ruby" "go" "python" 
for lang in "python"
do
	echo "processing language $lang"
	args="collection=codesearchnet benchmark=codesearchnet_corpus 
		searcher=BM25_reranker 
		searcher.k1='1.3'	
		searcher.b='1.0'  
		searcher.hits=1000 
		searcher.searcher.includetrain=False  
		searcher.searcher.benchmark.lang=$lang collection.lang=$lang benchmark.lang=$lang"
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

