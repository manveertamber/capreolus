do_train=$1
do_eval=$2
out_file="csn.ranking.scores"

echo "running BM25RM3"
# for lang in "go" "javascript" "java" "python" "php" # "ruby" "go" "python" 
for lang in "python" "php" # "ruby" "go" "python" 
do
	echo "processing language $lang"
	args="collection=codesearchnet benchmark=codesearchnet_corpus searcher=BM25RM3 
                searcher.k1='1.0-1.1-1.2' searcher.b='0.9-1.0' 
		searcher.fbTerms='55-75-95' searcher.fbDocs='2' searcher.originalQueryWeight='0.7-0.9'
		searcher.rerank=True
		searcher.hits=1000 searcher.searcher.includetrain=False  
		searcher.searcher.benchmark.remove_punc=False collection.remove_punc=False benchmark.remove_punc=False
		searcher.searcher.benchmark.lang=$lang collection.lang=$lang benchmark.lang=$lang"

	echo $args

	if $do_train 
	then
		echo "training"
		# python run.py rank.train with $args 
		# python run.py nonnn_rerank.train with $args 
		python run.py filter_rank.train with $args
	fi


	if $do_eval 
	then
		echo "evaluating"
		# python run.py rank.evaluate with $args 
		# python run.py nonnn_rerank.evaluate with $args 
		python run.py filter_rank.evaluate with $args
	fi
done

