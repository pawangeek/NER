# Word2vec Hindi

A skip-gram and CBOW Word2Vec model to create Hindi word embeddings on IIT Bombay Hindi monolingual corpus.

### Dataset

The dataset has been downloaded from the following link:
http://www.cfilt.iitb.ac.in/iitb_parallel/

### Skip-gram

In this model of Word2Vec, we use a single word from the text and try to predict it's context. Eg. for a sentence 'i love icecreams', our input would be 'love' and we would train the model to predict 'i' and 'icecreams'.

### Continuous Bag-of-Words (CBOW)

In this model of Word2Vec, we use the context words and try to predict our word from the context. Eg. for a sentence 'i love icecreams', our input would be ['i', 'icecreams'] and we would train the model to predict 'love'.