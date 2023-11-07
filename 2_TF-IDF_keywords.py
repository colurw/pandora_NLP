import pickle
import pandas as pd
import imports.c_TF_IDF as tfidf

TOP_K = [100, 300]

for NUM in TOP_K:

    # load list of positive-direction high-activation cluster_ids sorted by article index
    with open(f"pickles/dim_max_list_{NUM}.pkl", "rb") as file:
        dim_max_list = pickle.load(file)

    # read all csv data to pandas dataframe
    pd.options.mode.copy_on_write = True
    df = pd.read_csv("cleaned_data/abstracts_over150chars.txt", header=None, index_col=None)
    df.columns =["index", "title", "abstract"]
    print(df.head(30))

    # read cluster labels to pandas dataframe, concatenate with dataframe
    df_labels = pd.DataFrame({"cluster": dim_max_list})   
    df = pd.concat([df, df_labels], axis=1)
    print(df.head(10))

    # drop articles unneccesary columns
    df.drop(columns=["index", "title"], axis=1, inplace=True) 

    # aggregate abstracts by cluster label
    df_agg = df.groupby(["cluster"], as_index=False, sort=True).agg({"abstract": " ".join})
    print(df_agg.head(10))

    # calculate TF-IDF (text frequency-inverse document frequency) for each cluster against all clusters
    tf_idf, count = tfidf.c_tf_idf(documents=df_agg.abstract.values, 
                                m=len(df), 
                                ngram_range=(1, 1))    # length of keyword groups

    # create dictionary of minima keywords sorted by cluster_id
    max_keywords_dict = tfidf.extract_top_n_words_per_topic(tf_idf, count, 
                                                            docs_by_topic=df_agg, 
                                                            n=10)

    # save dictionary of keywords sorted by dimension
    with open(f"pickles/max_keywords_dict_{NUM}.pkl", "wb") as file:
        pickle.dump(max_keywords_dict, file)



    # load list of negative-direction high-activation cluster_ids sorted by article index
    with open(f"pickles/dim_min_list_{NUM}.pkl", "rb") as file:
        dim_min_list = pickle.load(file)

    # read all csv data to pandas dataframe
    pd.options.mode.copy_on_write = True
    df = pd.read_csv("cleaned_data/abstracts_over150chars.txt", header=None, index_col=None)
    df.columns =["index", "title", "abstract"]
    print(df.head(30))

    # read cluster labels to pandas dataframe, concatenate with dataframe
    df_labels = pd.DataFrame({"cluster": dim_min_list})   
    df = pd.concat([df, df_labels], axis=1)
    print(df.head(10))

    # drop articles unneccesary columns
    df.drop(columns=["index", "title"], axis=1, inplace=True) 

    # aggregate abstracts by cluster label
    df_agg = df.groupby(["cluster"], as_index=False, sort=True).agg({"abstract": " ".join})
    print(df_agg.head(10))

    # calculate TF-IDF (text frequency-inverse document frequency) for each cluster against all clusters
    tf_idf, count = tfidf.c_tf_idf(documents=df_agg.abstract.values, 
                                m=len(df), 
                                ngram_range=(1, 1))    # length of keyword groups

    # create dictionary of minima keywords sorted by cluster_id
    min_keywords_dict = tfidf.extract_top_n_words_per_topic(tf_idf, count, 
                                                            docs_by_topic=df_agg, 
                                                            n=10)

    # # save dictionary of keywords sorted by dimension
    with open(f"pickles/min_keywords_dict_{NUM}.pkl", "wb") as file:
        pickle.dump(min_keywords_dict, file)
        