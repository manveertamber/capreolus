do_train=$1
do_eval=$2
out_file="csn.ranking.scores"

echo "running BM25"
for lang in "php" "python" # "go" "javascript" "java"
do
	echo "processing language $lang"
	args="collection=codesearchnet benchmark=codesearchnet_corpus 
		searcher=BM25RM3 searcher.rerank=True 
		searcher.k1='1.0-1.1-1.2' searcher.b='0.9-1.0'
		searcher.fbTerms='55-75-95' searcher.fbDocs='2' searcher.originalQueryWeight='0.9'
		searcher.hits=1000 
		searcher.searcher.includetrain=False  
		searcher.searcher.benchmark.camelstemmer=False collection.camelstemmer=False benchmark.camelstemmer=False
		searcher.searcher.benchmark.lang=$lang collection.lang=$lang benchmark.lang=$lang"

	echo $args

	# searcher.k1=1.3~1.2~1.1~1.0~0.9~0.8 searcher.b=0.7~0.8~0.9~1.0  
	if $do_train 
	then
		echo "training"
		# python run.py rank.train with $args 
		python run.py filter_rank.train with $args
	fi


	if $do_eval 
	then
		echo "evaluating"
		python run.py filter_rank.evaluate with $args
	fi
done

