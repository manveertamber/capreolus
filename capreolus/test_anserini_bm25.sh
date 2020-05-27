do_train=$1
do_eval=$2

remove_punc=True

echo "running BM25"
# for lang in  "javascript" "php" "java" # "ruby" "go" "python" 
for lang in "ruby" "go" "javascript" # "python" 
do
	echo "processing language $lang"
	args="collection=codesearchnet benchmark=codesearchnet_corpus 
		searcher=BM25 
		searcher.k1='0.5-0.6-0.7-0.8-0.9-1.0-1.1-1.2-1.3-1.4' searcher.b='0.4-0.5-0.6-0.7-0.8-0.9-1.0'
		searcher.rerank=True searcher.hits=1000 
		searcher.searcher.benchmark.lang=$lang searcher.searcher.includetrain=False collection.lang=$lang benchmark.lang=$lang"
	
		
	# searcher.searcher.benchmark.camelstemmer=False collection.camelstemmer=False benchmark.camelstemmer=False 
	# searcher.fbTerms='70' searcher.fbDocs='15' searcher.originalQueryWeight='0.2'
	# searcher.k1='1.0' searcher.b='0.9'
	echo $args

	if $do_train 
	then
		echo "training"
		python run.py filter_rank.train with $args 
	fi


	if $do_eval 
	then
		echo "evaluating"
		python run.py filter_rank.evaluate with $args 
	fi
done

