# pandora_NLP

Deep learning models can be critised for taking a 'black box' approach to
modelling data.  Whilst we can judge the accuracy of the outputs, the lack of 
information available about the process that derives them leads to wariness 
about aspects of their use.  For example, studying whether discriminatory biases 
exist deep within the model can be difficult.

In some cases it might be possible to peek inside the box and understand what 
internal representions (concepts) the algorithm is using to make its inferences.
Ideally, this would mean understanding what each node is representing, and hence
its contribution to the network's final output.

## Bug Or Feature

In machine learning, each row of training data is represented by a feature
vector, which is simply a list of numbers.  Each number in the vector represents a point along a unique axis.

As a feature vector passes through a neural network, the number of its dimensions 
is typically altered upon arrival at each new layer, with one node in that layer
representing one new dimension.  The effect of this manipulation is the abstraction of the features beyond 
human comprehension, hence the black box problem.

## Embeddings As Node Activation States

A document-level NLP embedding is an example of a dataset processed to reduce 
dimensionality, where large one-hot encoded vectors representing individual words
in each document are compressed into smaller feature vectors, where each value in the vector representing
a concept (as understood by the computer) found in the document corpus.  

These 'cardinal' concepts are learned by the machine as the most efficient way to
discriminate between separate training documents, and may or may not make much sense to a 
human.

Embeddings are created by saving feature vectors emerging at the middle layer 
(i.e. the layer with the smallest number of nodes) of an autoencoder neural network.

The goal of this project is to describe the abstract concepts represented by these 
nodes/dimensions in a Universal Sentence Encoder model, by analysing the contents 
of documents that trigger very large activations (both +/-) of a particular node. 

## Python Scripts

#### 1_find_extreme_articles 

Searches through the embedded data for articles with the highest values in a particular
dimension of their feature vectors.  It creates a list of which, if any, dimension each
article in the dataset strongly represents.

#### 2_TF-IDF_keywords

Manipulates the original data to allow a TF-IDF (text frequency-inverse document 
frequency) calculation, that finds the most significant words (keywords) in the cluster
of documents found at the two extremes of each dimension in the embedding.

#### 3_chatGPT_dimension_labelling

Generates a list of strongly-representative article abstracts for each dimension and 
uses chat-gpt to infer the two concepts that best represent its positive and negative 
directions.  These concepts, along with keyword data, and article examples, are written
to a text file.

In an effort to eliminate the effects of noise in the data the keywords found in the 
top 100 articles for the dimension are compared to those of the top 300 articles.  Also
the concept labelling function is called twice per direction (_i.e._ four times per dimension),
to compare the results from the top 10 articles to the following 10 articles.  A similar/coherent
result between all of these data points should act as a validation step for any discovered concepts.

## Um... "Pandora", Yes?  Does It Work?

Overlooking the 'vengeful gods releasing evils into the world' aspects of the Pandora myth, and instead just focusing whether this 'box' gets opened, we can draw some conclusions.

Often the dimensions were found have a summarisable and validated concept in only one direction, meaning they demonstrate a cohesive group of keywords found in both the top 100 & top 300 articles, and which related to the two concepts described in the top 0-10 & top 11-20 articles.  

Typically in these cases, the other direction of the dimension will give much more varied (almost random) results, with their concept being labelled as "diverse".  Notably, when both directions of a dimension could be labelled with a degree of certainty, the concepts were unrelated, instead of being mutually opposite.

This was not always the case however, and many dimensions did not demonstrate an easily-determinable concept label in either direction.

It might be possible to improve on these results by not just searching for the most extreme articles for a given dimension, but instead finding articles of which the direction of its feature vector is more closely aligned with the dimension of interest - _i.e._ the remaining numbers in its feature vector are closer to zero, compared to the random numbers found at present.

The ChatGPT prompt could be improved/optimised, for example by including the results of the keyword analysis, or by connecting to a more recent (and costly) model, or simply by feeding it more than ten articles at a time.  And it should be remembered that ChatGPT, whilst possibly offering a more objective analysis than a human, is neither infallible, nor interested in checking its outputs for correctness.

It should also be noted that it is uncertain if the labels given represent fundamental underlying concepts, as their choice will be somewhat influenced by the random inital state of the model prior to its training.
