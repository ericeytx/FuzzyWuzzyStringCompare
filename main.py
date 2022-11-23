#!/usr/bin/env python
import pandas as pd
import numpy as np
from thefuzz import process, fuzz
import sqlite3

# Applying FuzzyWuzzy adapted from article by Thanh Huynh:
#   https://github.com/thuynh323/Natural-language-processing/blob/master/FuzzyWuzzy%20-%20Ramen%20Rater%20List/Find%20similar%20strings%20with%20FuzzyWuzzy.ipynb
# English dictionary for sample data sourced from EMDTDev/eddydn:
#   https://github.com/eddydn/DictionaryDatabase

# Install python-Levenshtein to avoid using slow pure-python SequenceMatcher
# pip3 install python-Levenshtein


# load English dictionary
con = sqlite3.connect("EDMTDictionary.db")
dictionary = pd.read_sql_query("SELECT * from WORD", con)
con.close()


# # verify dictionary data loaded into dataframe
# print(dictionary.head())

# asdljkas;ldfkjas;ldfkj


# get some general characteristics about the data set
for col in dictionary[['Word', 'Description']]:
    dictionary[col] = dictionary[col].str.strip()
    print('Number of unique values in ' + str(col) +': ' + str(dictionary[col].nunique()))


# get subset of unique descriptions from the dictionary
# we are going to compute Levenshtein distance/tokenize a cross-product of the input dataset
# for brevity of running shorter tests, sticking to a smaller sample (500^2 instead of 166,091^2...)
unique_description = dictionary['Description'].unique().tolist()[:500]


# create tuple of description comparisons sorted by similarity
score_sort = [(x,) + i
    for x in unique_description
    for i in process.extract(x, unique_description, scorer=fuzz.token_sort_ratio)]


# load tuple into similarity_sort dataframe
similarity_sort = pd.DataFrame(score_sort, columns=['string_sort','match_sort','score_sort'])
# print(similarity_sort.head())


# exclude reverse matches
similarity_sort['sorted_string_sort'] = np.minimum(similarity_sort['string_sort'], similarity_sort['match_sort'])
# print(similarity_sort.head())


# take samples passing match threshold (80%?)
high_score_sort = similarity_sort[(similarity_sort['score_sort'] >= 80) & 
    (similarity_sort['string_sort'] !=  similarity_sort['match_sort']) &
    (similarity_sort['sorted_string_sort'] != similarity_sort['match_sort'])]
high_score_sort = high_score_sort.drop('sorted_string_sort',axis=1).copy()


# present results
high_score_sort.groupby(['string_sort','score_sort']).agg(
                        {'match_sort': ', '.join}).sort_values(
                        ['score_sort'], ascending=False)
print(high_score_sort.head())
