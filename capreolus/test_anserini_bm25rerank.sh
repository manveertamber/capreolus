do_train=$1
do_eval=$2

echo "running BM25"
# for lang in  "javascript" "php" "java" # "ruby" "go" "python" 
for lang in "ruby"
do
	echo "processing language $lang"
	args="collection=codesearchnet benchmark=codesearchnet_corpus searcher=BM25RM3 
		searcher.k1='1.0' searcher.b='0.9'
		searcher.fbTerms='10' searcher.fbDocs='10-15' searcher.originalQueryWeight='0.5'
		searcher.rerank=True searcher.hits=1000 searcher.searcher.benchmark.lang=$lang searcher.searcher.includetrain=False  
		collection.lang=$lang benchmark.lang=$lang"

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

