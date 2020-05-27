do_train=$1
do_eval=$2
out_file="csn.ranking.scores"

echo "running BM25"
for lang in "go" "php" "ruby" "javascript" "python" "java"  
do
	echo "processing language $lang"
	# args="searcher=BM25 searcher.index.indexstops=True searcher.index.stemmer=none searcher.k1=1.2 searcher.b=0.75 searcher.hits=100 collection=codesearchnet_camel_parser benchmark=codesearchnet_corpus_camel_fix collection.lang=$lang benchmark.lang=$lang"
	# args="searcher=BM25 searcher.hits=100 collection=codesearchnet benchmark=codesearchnet_corpus collection.lang=$lang benchmark.lang=$lang searcher.k1=1.2 searcher.b=0.75"
	args="searcher=BM25 searcher.hits=100 collection=codesearchnet benchmark=codesearchnet_corpus collection.lang=$lang benchmark.lang=$lang"

	if $do_train 
	then
		echo "training"
		python run.py rank.train with $args 
	fi


	if $do_eval 
	then
		python contrib/filter_bm25_results.py -l $lang -o True --withcamel True \
			-csn "/home/xinyu1zhang/.capreolus/cache/collection-codesearchnet_camelstemmer-True_lang-%s/index-anserini_indexstops-False_stemmer-porter/searcher-BM25_b-0.4_hits-100_k1-0.9/codesearchnet_corpus/searcher" 
		echo "evaluating"
		python run.py rank.evaluate with $args 
	fi
done

