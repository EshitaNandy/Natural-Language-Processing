#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries and Dataset

# ## The first step is to import the libraries required to execute the scripts, along with the dataset. The following code imports the required libraries:

# In[1]:


from __future__ import print_function
import os
import nltk
import nltk.corpus
nltk.download('popular')
print(os.listdir(nltk.data.find("corpora")))


# ## The next step is to download the dataset. We will use Python's NLTK library to download the dataset. We will be using the Gutenberg Dataset, which contains 3036 English books written by 142 authors, including the "Hamlet" by Shakespeare.
# 
# The following script downloads the Gutenberg dataset and prints the names of all the files in the dataset.

# In[2]:


nltk.download('gutenberg')
from nltk.corpus import gutenberg
nltk.corpus.gutenberg.fileids()


# ## The file shakespeare-hamlet.txt contains raw text for the novel "Hamlet". To read the text from this file, the raw method from the gutenberg class can be used:
# Let's print the first 5000 characters from out dataset:

# In[3]:


hamlet_text = nltk.corpus.gutenberg.raw('shakespeare-hamlet.txt')
print(hamlet_text[:5000])


# # Tokenizing words 

# In[4]:


AI = """In computer science, artificial intelligence (AI), sometimes called machine intelligence, is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals."""


# In[5]:


type(AI)


# In[6]:


from nltk.tokenize import word_tokenize


# In[7]:


AI_tokens = word_tokenize(AI)
AI_tokens


# In[8]:


len(AI_tokens)


# ## Frequency of each word

# In[9]:


from nltk.probability import FreqDist
fdist = FreqDist()


# In[10]:


for word in AI_tokens:
    fdist[word.lower()]+=1
fdist


# In[11]:


fdist_top10 = fdist.most_common(10) ## frequency of first 10 words
fdist_top10


# In[12]:


from nltk.tokenize import blankline_tokenize
AI_blank = blankline_tokenize(AI)
len(AI_blank)


# 1 indicates how many paragraphs we have separated by a new line.

# ## Tokenization Types:
# 1. Bigrams: Tokens of two consecutive written words
# 2. Trigrams Tokens of three consecutive written words
# 3. Ngrams: Tokens of any no. of consecutive written words

# In[13]:


from nltk.util import bigrams,trigrams,ngrams


# In[14]:


string  = "Instead of worrying about what you cannot control, shift your energy to what you can create."
quotes_tokens = nltk.word_tokenize(string)
quotes_tokens


# In[15]:


quotes_bigrams = list (nltk.bigrams(quotes_tokens)) ##tokens in paired form
quotes_bigrams


# In[16]:


quotes_ngrams = list (nltk.ngrams(quotes_tokens,5)) ##tokens in paired form
quotes_ngrams


# # Stemming
# Normalize words into its base form or root form

# In[17]:


from nltk.stem import PorterStemmer
pst = PorterStemmer()
pst.stem("Having")


# In[18]:


words_to_stem=['give','giving','given','gave']
for words in words_to_stem:
    print(words+":"+pst.stem(words))


# In[19]:


from nltk.stem import LancasterStemmer
lst=LancasterStemmer()
for words in words_to_stem:
    print(words+":"+lst.stem(words))


# # Lemmatization
# - Groups together different inflected forms of a word called Lemma
# - Somehow similar to stemming, as it maps several words into one common root
# - Output of Lemmatization is a proper word

# In[20]:


from nltk.stem import wordnet
from nltk.stem import WordNetLemmatizer
word_len=WordNetLemmatizer()
word_len.lemmatize('corpora')


# In[21]:


words_to_lem=['gone', 'Going']
for words in words_to_lem:
    print(words+":"+word_len.lemmatize(words))


# # Stopwords

# In[22]:


from nltk.corpus import stopwords 
stopwords.words('english')


# In[23]:


len(stopwords.words('english'))


# In[24]:


fdist_top10


# In[25]:


import re
punctuation=re.compile(r'[-.?!,:;()\'|0-9]')
#compile function from re module to form a list with any digit or special charater


# In[26]:


post_punctuation=[]
for words in AI_tokens:
    word=punctuation.sub("/",words)
    if len(word)>0:
        post_punctuation.append(word)


# In[27]:


post_punctuation


# In[28]:


len(post_punctuation)


# # POS: Parts of speech
# - Generally speaking gramatical types of words like noun,verb,adverb, ajectives
# - A word can have more then one parts of speech based on the context it is used
# - Ex. "Google something on the internet", Here google is used as a verb although its a noun. 
# - These are some sort of ambiquities or difficulties which makes the NLU much more harder as compared to NLG
# - Because once u understand the language Generation is quite easy

# In[29]:


sent='krishna is a natural when it comes to singing'
sent_tokens=word_tokenize(sent)


# In[30]:


for token in sent_tokens:
    print(nltk.pos_tag([token]))


# # Named Entity Recognition
# Naming such as--
# - movie
# - monetary value
# - organization
# - location
# - quantities
# - person from a text

# ## Three phases of NAE
# 1.  Noun Pharase identification:-Extract all the noun phrases from a text using dependency passing and parts of speech tagging
# 2. Phase classification:- in this step all the extracted nouns and phrases are classified into respective categories such as location names and much more
# 3. Validation layer to evalute if something goes wrong using knowledge graphs</br>

# - Popular knowledge craft:- Google knowledge graph, IBM Watson, Wkipedia
# - Google's CEO Sundar Pichai introduced the new pixel at Minnesota Roi Centre Event
# - Google-Organization
# - Sundar Pichai- person
# - Minnesota-Location
# - Roi centre event-location

# In[31]:


import nltk
from nltk import ne_chunk
NE_sent="The Indian Politicians shouts in the Parliament House"


# In[32]:


NE_tokens=word_tokenize(NE_sent)
NE_tags=nltk.pos_tag(NE_tokens)


# In[33]:


NE_NER=ne_chunk(NE_tags)
print(NE_NER)


# # Chunking
# picking up individual pieces of information and grouping them into bigger pieces

# In[34]:


new = "The cat sat on a mat and ate the rat"
new_Tokens = nltk.pos_tag(word_tokenize(new))
new_Tokens


# In[35]:


grammar_np = r"NP:{<DT>?<JJ>*<NN>}"
chunk_parser = nltk.RegexpParser(grammar_np)


# In[36]:


chunk_result = chunk_parser.parse(new_Tokens)
chunk_result


# # ML Classifier - Movie Review from the nltk corpora

# In[37]:


import pandas as pd
import numpy as np


# In[38]:


from sklearn.feature_extraction.text import CountVectorizer
print (os.listdir(nltk.data.find("corpora")))


# In[39]:


from nltk.corpus import movie_reviews
print(movie_reviews.categories())


# In[40]:


print(len(movie_reviews.fileids('pos')))
print(' ')
print(movie_reviews.fileids('pos'))


# In[41]:


print(len(movie_reviews.fileids('neg')))
print(' ')
print(movie_reviews.fileids('neg'))


# In[42]:


neg_rev=movie_reviews.fileids('neg')
rev = nltk.corpus.movie_reviews.words('pos/cv000_29590.txt')
rev


# In[43]:


rev_list=[]
for rev in neg_rev:
    rev_text_neg=rev=nltk.corpus.movie_reviews.words(rev)
    review_one_string= " ".join(rev_text_neg)
    review_one_string=review_one_string.replace(' ,',',') # Remove space that comes with comma 
    review_one_string=review_one_string.replace(' .','.') # replace space that comes with dot
    rev_list.append(review_one_string)


# In[44]:


len(rev_list)


# In[46]:


pos_rev = movie_reviews.fileids('pos')
for rev_pos in pos_rev:
    rev_text_pos=rev=nltk.corpus.movie_reviews.words(rev_pos)
    review_one_string= " ".join(rev_text_neg)
    review_one_string=review_one_string.replace(' ,',',') # Remove space that comes with comma 
    review_one_string=review_one_string.replace(' .','.') # replace space that comes with dot
    rev_list.append(review_one_string)


# In[47]:


len(rev_list)


# In[48]:


neg_target=np.zeros((1000,),dtype=np.int)
pos_target=np.ones((1000,),dtype=np.int)


# In[49]:


target_list=[]
for neg_tar in neg_target:
 target_list.append(neg_tar)
for pos_tar in pos_target:
 target_list.append(pos_tar)


# In[50]:


len(target_list)


# In[51]:


y=pd.Series(target_list)


# In[52]:


type(y)


# In[53]:


y.head(20)


# In[54]:


from sklearn.feature_extraction.text import CountVectorizer


# In[55]:


count_vect=CountVectorizer(lowercase=True,stop_words='english',min_df=2)
x_count_vect=count_vect.fit_transform(rev_list)
x_count_vect.shape # will print (2000,16228)


# In[56]:


x_names=count_vect.get_feature_names()
x_names


# In[57]:


x_count_vect.shape

