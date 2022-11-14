import itertools
import re
import operator
import ast
import numpy as np
from sklearn.cluster import KMeans

with open("foods.txt", encoding='latin-1') as f:
    lines = f.readlines()

def getTop500Words():
    # Identify all the unique words that appear in the “review/text” field of the reviews.
    # Denote the set of such words as L.
    L = set()
    L_list = list()
    for line in lines:
        if line.startswith("review/text:"):
            words = line.split()
            for word in words:
                word = re.sub(r'\W+', '', word)
                L.add(word)
                L_list.append(word)
    # post-processing L list
    L.remove("reviewtext")
    L_list.remove("reviewtext")

    # Remove from L all stopwords in “Long Stopword List” from http://www.ranks.nl/
    # stopwords. Denote the cleaned set as W .
    with open('stop_words.txt') as f1:
        lines1 = f1.readlines()
    stopWords = set()
    for line in lines1:
        stopWords.add(line.strip())
    W = L - stopWords

    # Count the number of times each word in W appears among all reviews (“review/text” field)
    # and identify the top 500 words.
    wordCounts = {}
    for word in L_list:
        if word not in W:
            continue
        else:
            if word not in wordCounts:
                wordCounts[word] = 0
            else:
                wordCounts[word]=wordCounts.get(word)+1
    sortedCounts = {k: v for k, v in sorted(wordCounts.items(), key=operator.itemgetter(1), reverse=True)}
    top500 = dict(itertools.islice(sortedCounts.items(), 500))
    f = open("top500Words.txt","w")
    f.write( str(top500) )
    f.close()
#getTop500Words()

# Vectorize all reviews (“review/text” field) using these 500 words (see an example of vec-
# torization here: https://bit.ly/3CcY9i4).
with open("top500Words.txt") as f2:
    top500 = f2.read()
TopWordsDict = ast.literal_eval(top500)
EmbeddingDict = {}
i = 0
for k in TopWordsDict.keys():
    EmbeddingDict[k] = i
    i+=1

def tokenization(line):
    wordsList = list()
    words = line.split()
    for word in words:
        word = re.sub(r'\W+', '', word)
        wordsList.append(word)
    return wordsList

def vectorization(wordsList):
    vector = np.zeros(500)
    for word in wordsList:
        if word in EmbeddingDict:
            d_k = EmbeddingDict[word]
            vector[d_k]=vector[d_k]+1
    return vector

def getLineVectorRepresentation():
    lineVectors = []
    for line in lines:
        if line.startswith("review/text:"):
            wordsList = tokenization(line)
            lineVectors.append(vectorization(wordsList))
    lineVectors = np.array(lineVectors)
    np.save("line_vectors", lineVectors)
#getLineVectorRepresentation()

# Cluster the vectorized reviews into 10 clusters using k-means. You are allowed to use any
# program or code for k-means. This will give you 10 centroid vectors.
lineVectors = np.load('line_vectors.npy')
#print(lineVectors.shape) #(568454, 500)
kmeans = KMeans(n_clusters=10, random_state=0).fit(lineVectors)
print("------------ Cluster Centroid ------------")
print(kmeans.cluster_centers_)

# From each centroid, select the top 5 words that represent the centroid (i.e., the words with
# the highest feature values)
def get_key(val):
    for key, value in EmbeddingDict.items():
        if val == value:
            return key
    return "key doesn't exist"

for centroid in kmeans.cluster_centers_:
    top5index = np.argpartition(centroid, -5)[-5:]
    output = "Top 5 words: "
    for index in top5index:
        output = output + get_key(index) + " "
    print(output)
