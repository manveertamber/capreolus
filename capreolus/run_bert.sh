args="
  rank.searcher.name=msmarcopsg_searcher
  benchmark.name=msmarcopsg

  threshold=100
  optimize=ndcg_cut_20

  reranker.name=TFBERTMaxP
  reranker.extractor.usecache=True
  reranker.extractor.maxseqlen=256
  reranker.extractor.numpassages=30
  reranker.extractor.passagelen=150
  reranker.extractor.stride=75
  reranker.extractor.prob=0.1

  reranker.trainer.name=tensorflow 
  reranker.trainer.niters=36
  reranker.trainer.itersize=256
  reranker.trainer.validatefreq=2
  reranker.trainer.batch=8

  reranker.trainer.lr=0.001
  reranker.trainer.bertlr=0.00001
  reranker.trainer.warmupsteps=8
  reranker.trainer.warmupbert=False
  reranker.trainer.warmupnonbert=True

  reranker.trainer.decay=0.1
  reranker.trainer.decaystep=10
  reranker.trainer.decaytype=linear

  reranker.trainer.loss=pairwise_hinge_loss
  sampler.name=triplet

	reranker.trainer.name=tensorflow
	reranker.trainer.tpuzone=us-central1-f
	reranker.trainer.storage=gs://robust04/aaai
"

train=$1
eval=$2

#if $train; then
#  python run.py rerank.train with $args
#fi

#if $eval; then
#  python run.py rerank.evaluate with $args
#fi

for i in 1 # 2 3 4 5
do
python run.py rerank.train with $args reranker.trainer.tpuname=nodev2-$i  fold=s$i 
done

exit
for i in 2 3 4 5
do
nohup python run.py rerank.train with $args reranker.trainer.tpuname=nodev2-$i  fold=s$i &  
done

