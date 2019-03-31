
# coding: utf-8

# In[1]:

import pandas as pd
time_sentences = ["Monday: The doctor's appointment is at 2:45pm.", 
                  "Tuesday: The dentist's appointment is at 11:30 am.",
                  "Wednesday: At 7:00pm, there is a basketball game!",
                  "Thursday: Be back home by 11:15 pm at the latest.",
                  "Friday: Take the train at 08:10 am, arrive at 09:00am."]
df = pd.DataFrame(time_sentences, columns = ['text'])
df


# In[2]:

# find the number of characters for each string in df['text']
df['text'].str.len()


# In[3]:

# find the number of tokens for each string in df['text']
df['text'].str.split().str.len()


# In[4]:

# find which entries contain the word 'appointment'
df['text'].str.contains('appointment')


# In[5]:

# find how many times a digit occurs in each string
df['text'].str.count(r'\d')


# In[6]:

# find all occurances of the digits
df['text'].str.findall(r'\d')


# In[7]:

# group and find the hours and minutes
df['text'].str.findall(r'(\d?\d):(\d\d)')


# In[10]:

df['text'].str.replace(r'\w+day\b', '???')


# In[12]:

df['text'].str.replace(r'(\w+day\b)', lambda x: x.groups()[0][:3])


# In[13]:

# create new columns from first match of extracted groups
df['text'].str.extract(r'(\d?\d):(\d\d)')


# In[15]:

# extract the entire time, the hours, the minutes, and the period
df['text'].str.extractall(r'((\d?\d):(\d\d) ?([ap]m))')


# In[16]:

df['text'].str.extractall(r'(?P<time>(?P<hour>\d?\d):(?P<minute>\d\d) ?(?P<period>[ap]m))')

