import csv
import pickle
import openai
from num2words import num2words
from api_keys import secret
import textwrap
from time import sleep


def get_dimension_keywords(keywords_dict, dim, top_k):
    """ returns top_k keywords from a given nodes high-activation article cluster """
    keywords = []
    for keyword, score in keywords_dict[dim][0:top_k]:
        keywords.append(keyword)
    
    return keywords


def chatGPT_dimension_label(abstracts):
    """ uses a list of titles and abstracts as basis for a chatgpt query """
    openai.api_key = secret.key
    completion = openai.ChatCompletion()
    length = len(abstracts)

    # convert abstracts list into string with XML tags
    abstracts = ["<article> "+ abstract +" </article> \n" for abstract in abstracts]
    abstracts = " ".join(abstracts)
    
    # create query
    query = [{"role"   : f"system",
              "content": f"""You are an intelligent but laconic robot that responds with one sentence at all times.
                             You will be provided with {num2words(length+1)} articles delimited with XML tags. 
                             Your task is to label the overarching concept that a group of loosely related articles represents.  
                             Your concept label should describe some abstract aspect that is shared by every article in the group. 
                             Or, your concept label should describe the broad theme of any characteristics shared by the articles.
                             Your concept label should be as short as possible without ignoring any articles."""}, 
             {"role"   : f"user", 
              "content": f"""Hello robot, here are the {num2words(length+1)} articles: \n{abstracts}\n 
                             What concept label best represents the shared nature of these seemingly disparate articles?"""}]
    
    # get response
    try:
        response = completion.create(model="gpt-3.5-turbo", messages=query)   # OR model="gpt-4"
    except:
        sleep(30)
        print('retrying...')
        try:
            response = completion.create(model="gpt-3.5-turbo", messages=query)   # OR model="gpt-4"
        except:
            sleep(30)
            print('retrying again...')
            try:
                response = completion.create(model="gpt-3.5-turbo", messages=query)   # OR model="gpt-4"
            except:
                print('server too busy!')
    
    answer = response.choices[0]["message"]["content"]
    return answer


def get_article(articles, index):
    """ return title-abstract raw_articles for a given index """
    for row in articles:
        if int(row[0]) == int(index):
            title = str(row[1])
            abstract = str(row[2])
            print(row[0], title.upper())
            print(f" {abstract}")
            
            return title, abstract


def fprint(file, my_string, width=150, chop=False, enter=True):
    """ takes a string and prints it as a wrapped paragraph to a text file """
    if chop == True:
        my_string = my_string[0:width-3]+"..."
    
    if len(my_string) > width:
        my_string = textwrap.fill(my_string, width, fix_sentence_endings=True)
    
    with open(f"{file}.txt", "a") as file:
        if enter == True:
            file.write(my_string + "\n")
        else:
            file.write(my_string)        
    
    return my_string


# # delay running until USA goes to bed
# print("waiting...")
# sleep(12000)

# extract index-title-abstract data from cleaned csv file to list
raw_articles = []
with open("cleaned_data/abstracts_over150chars.txt") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader:
        raw_articles.append(row)

# load lists of high_activation dimension_ids sorted by article index
with open("pickles/dim_max_list_300.pkl", "rb") as file:
    dim_max_list = pickle.load(file)

with open("pickles/dim_min_list_300.pkl", "rb") as file:
    dim_min_list = pickle.load(file)

# load dictionaries of keywords sorted by dimension
with open("pickles/max_keywords_dict_100.pkl", "rb") as file:
    max_keywords_dict_100 = pickle.load(file)

with open("pickles/min_keywords_dict_100.pkl", "rb") as file:
    min_keywords_dict_100 = pickle.load(file)

with open("pickles/max_keywords_dict_300.pkl", "rb") as file:
    max_keywords_dict_300 = pickle.load(file)

with open("pickles/min_keywords_dict_300.pkl", "rb") as file:
    min_keywords_dict_300 = pickle.load(file)


# generate outputs
for dimension in range(0, 512):

    # generate list of indices of high-activation articles for the current dimensoin
    maxima = [index for index, dimension_id in enumerate(dim_max_list) if dimension_id == dimension]
    print(maxima)
    minima = [index for index, dimension_id in enumerate(dim_min_list) if dimension_id == dimension]
    print(minima)

    # generate list of 20 article abstracts from list of article indices
    max_abstracts = [get_article(raw_articles, index)[1] for index in maxima[0:20]]
    min_abstracts = [get_article(raw_articles, index)[1] for index in minima[0:20]]
    
    # analyse concept that best represents the shared aspects of the top 10 articles
    max_label = chatGPT_dimension_label(max_abstracts[0:10])
    min_label = chatGPT_dimension_label(min_abstracts[0:10])
    # check concept roughly matches second set of the next 10 articles
    max_label_validation = chatGPT_dimension_label(max_abstracts[10:20])
    min_label_validation = chatGPT_dimension_label(min_abstracts[10:20])
    
    # get keywords for current dimension
    max_keywords_100 = get_dimension_keywords(max_keywords_dict_100, dim=dimension, top_k=6)
    min_keywords_100 = get_dimension_keywords(min_keywords_dict_100, dim=dimension, top_k=6)
    max_keywords_300 = get_dimension_keywords(max_keywords_dict_300, dim=dimension, top_k=6)
    min_keywords_300 = get_dimension_keywords(min_keywords_dict_300, dim=dimension, top_k=6)

    # create output text file
    file = "results/dim_"+str(dimension+1).zfill(3) 
    fprint(file, "\nNODE / DIMENSION  " + str(dimension+1) + " of 512 \n")

    # print postive direction data
    fprint(file, "  POSITIVE DIRECTION \n")
    fprint(file, f"\t Keywords (100):  {str(max_keywords_100)}")
    fprint(file, f"\t Keywords (300):  {str(max_keywords_300)}")
    fprint(file, f"\t Concept:         {str(max_label)}")
    fprint(file, f"\t Concept (val):   {str(max_label_validation)} \n")
    # print list of truncated article abstracts
    fprint(file, "\t Extreme Articles:")
    for index in maxima[0:10]:
        fprint(file, f"\t  {raw_articles[int(index)][2]}", width=85, chop=True)

    # print negative direction data
    fprint(file, "\n\n  NEGATIVE DIRECTION \n")
    fprint(file, f"\t Keywords (100): {str(min_keywords_100)}")
    fprint(file, f"\t Keywords (300): {str(min_keywords_300)}")
    fprint(file, f"\t Concept:        {str(min_label)}")
    fprint(file, f"\t Concept (val):  {str(min_label_validation)} \n")
    # print list of truncated article abstracts
    fprint(file, "\t Extreme Articles:")
    for index in minima[0:10]:
        fprint(file, f"\t  {raw_articles[int(index)][2]}, ", width=85, chop=True)

    # write labels only to separate text file
    file = "results/~summary"
    fprint(file, f"NODE/DIM {str(dimension+1).zfill(3)} +ve.   {str(max_label)}  OR  {str(max_label_validation)} ", width=300)
    fprint(file, f"NODE/DIM {str(dimension+1).zfill(3)} -ve.   {str(min_label)}  OR  {str(min_label_validation)} \n", width=300)
