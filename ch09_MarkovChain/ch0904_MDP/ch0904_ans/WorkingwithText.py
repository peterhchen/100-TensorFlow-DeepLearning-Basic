#!/usr/bin/env python
# coding: utf-8

# In[1]:


text1 = "Ethics are built right into the ideals and objectives of the United Nations "

len(text1) # The length of text1


# In[2]:


text2 = text1.split(' ') # Return a list of the words in text2, separating by ' '.

len(text2)


# In[3]:


text2


# In[4]:


# List comprehension
[w for w in text2 if len(w) > 3] # Words that are greater than 3 letters long in text2


# In[5]:


[w for w in text2 if w.istitle()] # Capitalized words in text2


# In[6]:


[w for w in text2 if w.endswith('s')] # Words in text2 that end in 's'


# In[7]:


# find unique words using set()
text3 = 'To be or not to be'
text4 = text3.split(' ')

len(text4)


# In[8]:


len(set(text4))


# In[9]:


set(text4)


# In[10]:


len(set([w.lower() for w in text4])) # .lower converts the string to lowercase.


# In[11]:


set([w.lower() for w in text4])


# In[12]:


# Processing free-text
text5 = '"Ethics are built right into the ideals and objectives of the United Nations" #UNSG @ NY Society for Ethical Culture bit.ly/2guVelr'
text6 = text5.split(' ')

text6


# In[13]:


# finding hashtags
[w for w in text6 if w.startswith('#')]


# In[14]:


# finding callouts
[w for w in text6 if w.startswith('@')]


# In[15]:


text7 = '@UN @UN_Women "Ethics are built right into the ideals and objectives of the United Nations" #UNSG @ NY Society for Ethical Culture bit.ly/2guVelr'
text8 = text7.split(' ')


# In[16]:


import re # import re - a module that provides support for regular expressions

[w for w in text8 if re.search('@[A-Za-z0-9_]+', w)]


# In[ ]:




