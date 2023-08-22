#!/usr/bin/env python
# coding: utf-8

# In[14]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[15]:


# Sample dataset of songs and their genres
songs = [
    "Song A: Pop, Rock",
    "Song B: Pop",
    "Song C: Rock",
    "Song D: Jazz",
    "Song E: Pop, Jazz",
    "Song F: Rock, Jazz",
]


# In[16]:


# User input: the user likes songs with these genres
user_likes = "Pop"


# In[17]:



# Create a TF-IDF vectorizer to convert genre text into numerical vectors
tfidf_vectorizer = TfidfVectorizer()


# In[18]:


# Fit and transform the song genres into TF-IDF vectors
tfidf_matrix = tfidf_vectorizer.fit_transform(songs)


# In[19]:


# Compute the cosine similarity between the user's preference and each song
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_vectorizer.transform([user_likes]))


# In[20]:


# Sort the songs by their similarity score
similarity_scores = list(enumerate(cosine_sim))
similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)


# In[21]:



# Get the top recommendations
num_recommendations = 2  # You can change this to get more recommendations
top_recommendations = similarity_scores[1:num_recommendations + 1]


# In[22]:


# Print the recommended songs
print(f"Recommended Songs based on your preference for '{user_likes}':")
for i, score in top_recommendations:
    print(songs[i])


# In[ ]:





# In[ ]:




