import requests
from bs4 import BeautifulSoup
import streamlit as st
import pandas as pd
import numpy as np
import nltk
import contractions
import re 
import nltk
nltk.download('stopwords')
stopword_list=nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')
len(stopword_list)
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer=ToktokTokenizer()
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
vs=SentimentIntensityAnalyzer()
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
 
def html_tag(text):
  soup=BeautifulSoup(text,"html.parser")
  news_text=soup.get_text()
  return news_text 
 
def con(text):
  expand=contractions.fix(text) 
  return expand
 
def remove_sp(text):
  pattern=r'[^A-Za-z0-9\s]'
  text=re.sub(pattern,'',text) 
  return text
 
def remove_stopwords(text):
  tokens=tokenizer.tokenize(text)
  tokens=[token.strip() for token in tokens]
  filtered_tokens=[token for token in tokens if token not in stopword_list]
  filtered_text=' '.join(filtered_tokens) 
  return filtered_text
 
df = pd.read_csv('news.csv')
 
df.news_headline=df.news_headline.apply(lambda x:x.lower())
df.news_article=df.news_article.apply(lambda x:x.lower()) 
df.news_headline=df.news_headline.apply(html_tag)
df.news_article=df.news_article.apply(html_tag)
 
df.news_headline=df.news_headline.apply(con)
df.news_article=df.news_article.apply(con)
df.news_headline=df.news_headline.apply(remove_sp)
df.news_article=df.news_article.apply(remove_sp)
df.news_headline=df.news_headline.apply(remove_stopwords)
df.news_article=df.news_article.apply(remove_stopwords)
df['compound']=df['news_article'].apply(lambda x: vs.polarity_scores(x)['compound'])
x = df.iloc[:,1].values # Message column as input
y = df.iloc[:,0].values # Label column as output
 
st.title('SENTIMENT ANALYSIS USING PYTHON')
st.subheader('PREDICTS DATA HOW MUCH IT WILL HAVE POSITIVITY , NEGATIVITY AND NUETRAL')
 
text_model=Pipeline([('tfidf',TfidfVectorizer()),('model',MultinomialNB())])
text_model.fit(x,y)
message = st.text_area("Enter Text")
text_model.predict([message]) 
 
if st.button("Predict"):
  msg=html_tag(message.lower())
  msg=con(msg) 
  msg=remove_sp(msg) 
  msg=remove_stopwords(msg) 
  st.subheader('OUTPUT') 
  st.text(msg) 
  text=vs.polarity_scores(msg) 
  st.subheader('THE POLARITY SCORE IS:') 
  st.title(text)
