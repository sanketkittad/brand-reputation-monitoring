from flask import Flask,render_template
from flask import request
app=Flask(__name__)
import re 

import numpy as np


from textblob import TextBlob 

import matplotlib.pyplot as plt

import pandas as pd

from wordcloud import WordCloud

from better_profanity import profanity
@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/data",methods=['POST'])
def getData():
    if request.method=='POST':
        apple_tweets=pd.read_csv("apple_tweets.csv")
        samsung_tweets=pd.read_csv("samsung_tweets.csv")
        
        apple_tweets=apple_tweets[['tweet_id','username','full_text']]
        samsung_tweets=samsung_tweets[['tweet_id','username','full_text']]
        apple_list=apple_tweets['full_text'].to_list()
        samsung_list=samsung_tweets['full_text'].to_list()
        cleaned_apple=[clean(tweet) for tweet in apple_list]
        cleaned_samsung=[clean(tweet) for tweet in samsung_list]
        apple_objects=[TextBlob(tweet) for tweet in cleaned_apple]
        samsung_objects=[TextBlob(tweet) for tweet in cleaned_samsung]
        sentiment_apple = [[tweet.sentiment.polarity, str(tweet)] for tweet in apple_objects]
        sentiment_samsung = [[tweet.sentiment.polarity, str(tweet)] for tweet in samsung_objects]
        sentiment_df_apple = pd.DataFrame(sentiment_apple, columns=["polarity", "tweet"])
        sentiment_df_samsung = pd.DataFrame(sentiment_samsung, columns=["polarity", "tweet"])
        conditions =[(sentiment_df_apple['polarity'] >0.2),(sentiment_df_apple['polarity'] <0.2)]
        choices = ['Positive', 'Negative']
        sentiment_df_apple['sentiment'] = np.select(conditions, choices, default='Neutral')
        conditions =[(sentiment_df_samsung['polarity'] >0.2),(sentiment_df_samsung['polarity'] <0.2)]
        choices = ['Positive', 'Negative']
        sentiment_df_samsung['sentiment'] = np.select(conditions, choices, default='Neutral')
        samsung_pos=sentiment_df_samsung['sentiment'].value_counts()['Positive']
        samsung_neg=sentiment_df_samsung['sentiment'].value_counts()['Negative']
        samsung_neu=sentiment_df_samsung['sentiment'].value_counts()['Neutral']
        samsung_data={
            'sentiment':['Positive',
            'Negative',
            'Neutral'],'count':[samsung_pos,samsung_neg,samsung_neu]
        }
        samsung_data=pd.DataFrame(samsung_data)
        samsung_data=samsung_data.to_json()
        
        apple_pos=sentiment_df_apple['sentiment'].value_counts()['Positive']
        apple_neg=sentiment_df_apple['sentiment'].value_counts()['Negative']
        apple_neu=sentiment_df_apple['sentiment'].value_counts()['Neutral']
        apple_data={
            'sentiment':['Positive',
            'Negative',
            'Neutral'],'count':[apple_pos,apple_neg,apple_neu]
        }
        apple_data=pd.DataFrame(apple_data)
        apple_data=apple_data.to_json()
        return render_template(template_name_or_list='main.html',samsung_data=samsung_data,apple_data=apple_data)
    


def clean(tweet):
    if(type(tweet)==float):
        return ""
    r=tweet.lower()
    r=profanity.censor(r)
    r = re.sub("'", "", r) # This is to avoid removing contractions in english
    r = re.sub("@[A-Za-z0-9_]+","", r)
    r = re.sub("#[A-Za-z0-9_]+","", r)
    r = re.sub(r'http\S+', '', r)
    r = re.sub('[()!?]', ' ', r)
    r = re.sub('\[.*?\]',' ', r)
    r = re.sub("[^a-z0-9]"," ", r)
    r = r.split()
    stopwords = ["for", "on", "an", "a", "of", "and", "in", "the", "to", "from"]
    r = [w for w in r if not w in stopwords]
    r = " ".join(word for word in r)
    return r

