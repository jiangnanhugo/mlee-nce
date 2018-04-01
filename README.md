# mlee-nce
Biological Event Trigger Identification with Noise Contrastive Estimation. IEEE/ACM TRANSACTIONS ON COMPUTATIONAL BIOLOGY AND BIOINFORMATICS, 2017.

1. keras

   use keras to inplement the event trigger identification pipeline.

2. theano

   use theano to implement the event trigger identification pipeline, the training data is reconstruct by NCE algorithm. If the sample size is set to $k$, then the training batch size should be set to $k+1$, so that this training process is align to the algorithm in paper.

3. word-embedding

   perl script is used for download pubmed articles, along with XML extractor. Also you can use the `gensim.Word2Vec` for training the word-embedding features.

   ​