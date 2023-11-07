import numpy as np
import copy
import pickle


def find_extremes(array, dimension, top_k):
    """ returns two lists of vector indices with top_k most extreme feature values in a given dimension """
    arr = copy.deepcopy(array)
    maxima = []
    minima = [] 
    for k in range(top_k):
        print(dimension, k)
        # find index of row with extreme feature value
        index_h = arr[ : , dimension].argmax()
        index_l = arr[ : , dimension].argmin()
        maxima.append(index_h)
        minima.append(index_l)
        # set rows feature vector to zero to prevent double counting
        arr[index_h] = np.zeros(arr.shape[1])
        arr[index_l] = np.zeros(arr.shape[1])
    
    return maxima, minima


TOP_K = [100, 300]  # number of extreme articles to find per dimension of embedding

for NUM in TOP_K:

    # load embedded articles
    with open("models/USE/over150chars/all_embeddings.pkl", mode="rb") as file:
        embeddings = pickle.load(file)
    print(embeddings.shape)

    # generate blank list of cluster_ids sorted by index, one for each article in embedding
    dim_max_list = [-1 for article in range(embeddings.shape[0])]
    dim_min_list = [-1 for article in range(embeddings.shape[0])]

    for dimension in range(0, 512):
        # find extreme values for a given dimension
        maxima, minima = find_extremes(embeddings, dimension, top_k=NUM)
        
        # update list with dimension_ids of high-positive-activation articles
        for index in maxima:
            dim_max_list[index] = dimension
        
        # update list with dimension_ids of high-negative-activation articles
        for index in minima:
            dim_min_list[index] = dimension

    # save list of high_activation cluster_ids sorted by article index
    with open(f"pickles/dim_max_list_{NUM}.pkl", "wb") as file:
        pickle.dump(dim_max_list, file)

    with open(f"pickles/dim_min_list_{NUM}.pkl", "wb") as file:
        pickle.dump(dim_min_list, file)