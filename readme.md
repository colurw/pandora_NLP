# pandora_NLP

<img src="https://github.com/colurw/pandora_NLP/assets/66322644/aa26371a-0df5-4321-9e2f-4bc0b72a3f65" align="right" width="450px"/>
Deep learning models can be criticised for taking a 'black box' approach to
modelling data.  Whilst one can judge the accuracy of their outputs, the lack of 
information available about the process that derives them leads to wariness 
about some aspects of their use.  For example, studying whether discriminatory biases 
exist deep within a model can be difficult.

In some cases, it might be possible to peek inside this box and understand what 
latent feature representations (concepts) the algorithm is using to make its inferences.
Ideally, this would mean understanding what each network node represents in the latent space. <br clear="right"/>

## Bug Or Feature

In machine learning, each row of training data is represented by a feature
vector (a list of numbers).  Each number in the vector can be thought of as point along a unique axis/dimension.

As a feature vector passes through a neural network, the number of its dimensions 
is often altered by each new layer, with each node in that layer
representing one dimension.  The effect of this is the abstraction of latent features (meaning they cannot be directly observed) - hence the black box problem.




## Embeddings As Node Activation States

A document-level NLP embedding is an example of a dataset processed to reduce 
dimensionality, where large one-hot encoded vectors representing individual words
in each document are compressed into smaller feature vectors, where each value in the vector represents
a concept (as understood by the computer) in the document.  

These 'cardinal' concepts are learned by the machine as the most efficient way to
distinguish between individual training documents.

Embeddings are created by saving feature vectors emerging at the middle layer 
(_i.e._, the layer with the fewest nodes) of an autoencoder neural network.

The aim of this project is to describe the abstracted concepts represented by these 
nodes/dimensions in a Universal Sentence Encoder (USE) model, by analysing the contents 
of documents that trigger the largest activations (both +/-) of a particular node. 

The documents used are the dataset of 1.2 million first paragraphs of Wikipedia pages,
created by github.com/colurw/wiki_abstracts_NLP

## Python Scripts

#### 1_find_extreme_articles 

Searches through the embedded data for articles with the highest values in a particular
dimension of their feature vector.  It creates a list of which dimension each
article in the dataset strongly activates, with placeholders ('-1') if the article
is not one of the most extreme articles in any dimension.  

#### 2_TF-IDF_keywords

Manipulates the original data to allow a text frequency-inverse document 
frequency (TF-IDF) calculation, that finds the most significant words (keywords) in the cluster
of documents found at the two extremes of each dimension in the embedding.

#### 3_chatGPT_dimension_labelling

Generates a list of strongly-representative article abstracts for each dimension and 
uses ChatGPT to infer the two concepts that best represent its positive and negative 
directions.  These concepts, along with keyword data, and article examples, are written
to text files.

In an effort to eliminate the effects of noise in the data, the keywords found in the 
top 100 articles for the dimension are compared to those of the top 300 articles.  Also,
the concept labelling function is called twice per direction (four times per dimension)
to compare the top 10 articles to the following 10 articles.  A similar/coherent
result acting as a (simplistic) validation of any discovered concepts.

## Um... "Pandora", Yes?  Does It Work?

Overlooking the 'vengeful gods releasing evils into the world' aspects, and instead just focusing whether 'the box' gets opened, we can draw some conclusions.

Often the dimensions were found to have a surmisable and 'validated' concept in only one direction. Typically, in these cases, the other direction of the dimension will give much more varied (almost random) results, with their two concepts being labelled as "diverse" or "miscellaneous" or having no obvious similarity.  Notably, when opposite ends of a dimension could be labelled with a degree of certainty, their concept labels were unrelated ("Film and entertainment / Performing Arts" _vs._ "Exploration / Historical Figures and Events"), rather than being mutual opposites. (_E.g._ "Childhood / Children's Education" _vs._ "Death and its various aspects / Death").

This was not always the case however, and many dimensions did not demonstrate an easily-determinable concept label in either direction.

It might be possible to improve on these results by not just searching for the most extreme articles for a given dimension, but instead finding articles for which the direction of its feature vector (in multi-dimensional space) is more closely aligned with the dimension of interest - _i.e._, the other numbers in its feature vector are closer to zero, compared to the (normally distributed) random numbers found at present.

The ChatGPT prompt could be optimised, for example by improving the system prompt, including the results of the keyword analysis, by connecting to a more recent (and costly) model, or simply by feeding it more than ten articles at a time.  
