# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 12:28:01 2020

@author: Lenovo
"""

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import API
from tweepy import Stream
from tweepy import Cursor
import twitter_credentials
import numpy as np
import pandas as pd
import re
import dash
import plotly.graph_objs as go
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import json
import tweepy 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
import dash_table
from wordcloud import WordCloud
from io import BytesIO
import base64
import webbrowser
from threading import Timer

stop_words = set(stopwords.words('english')) 
                 

class TwitterClient():
    def __init__(self,twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client=API(self.auth)
        self.twitter_user=twitter_user
    def get_timeline_tweets(self,n_tweets):
        tweets=[]
        for tweet in Cursor(self.twitter_client.user_timeline,id=self.twitter_user).items(n_tweets):
            tweets.append(tweet)
        return tweets
    def get_twitter_client_api(self):
        return self.twitter_client
class TwitterListner(StreamListener):
    
    def __init__(self,file_nm):
        self.file_nm=file_nm
    
    def on_data(self,data):
        try:
            print(data)
            with open(self.file_nm,'w') as tf:
                tf.write(json.dumps(data)+"\n")
            return True
        except BaseException as e:
            print("Error: %s" % str(e))
        return True
    def on_error(self,status):
        if(status==420):
            return False
        print(status)   

class MyStreamListener(tweepy.StreamListener):
    """
    Twitter listener, collects streaming tweets and output to a file
    """
    def __init__(self, api=None):
        super(MyStreamListener, self).__init__()
        self.num_tweets = 0
        self.file = open("tweets.txt", "w")

    def on_status(self, status):
        tweet = status._json
        self.file.write( json.dumps(tweet) + '\n' )
        self.num_tweets += 1
        
        # Stops streaming when it reaches the limit
        if self.num_tweets <= 200:
            if self.num_tweets % 100 == 0: # just to see some progress...
                print('Numer of tweets captured so far: {}'.format(self.num_tweets))
            return True
        else:
            return False
        self.file.close()

    def on_error(self, status):
        print(status)

class TwitterAuthenticator():
    
    def authenticate_twitter_app(self):
        auth = OAuthHandler(twitter_credentials.Consumer_key, twitter_credentials.Consumer_key_secret)
        auth.set_access_token(twitter_credentials.Access_key, twitter_credentials.Access_key_secret)
        return auth
class TweetStreamer():
    
    def __init__(self):
        self.twitter_authenticator = TwitterAuthenticator()
    def stream_tweets(self,search):
        try:
            listener = MyStreamListener()
            auth=self.twitter_authenticator.authenticate_twitter_app()
            stream=Stream(auth, listener)
            stream.filter(track=[search],languages=['en'])
        except KeyboardInterrupt:
            print("Streaming stopped.Processing further tasks......")
class TweetAnalysis():
    def clean_tweet(self,tweet):
                tweet.lower()
                tweet = re.sub(r"http\S+", '', tweet, flags=re.MULTILINE)
                tweet = re.sub(r'\@\w+|\#','', tweet)
                tweet = tweet.translate(str.maketrans('', '', string.punctuation))
                tweet_tokens = word_tokenize(tweet)
                filtered_words = [w for w in tweet_tokens if not w in stop_words]
    
                ps = PorterStemmer()
                stemmed_words = [ps.stem(w) for w in filtered_words]
                lemmatizer = WordNetLemmatizer()
                lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]
                return " ".join( lemma_words)
    
    def analyze_sentiment(self,tweet):
        analysis = SentimentIntensityAnalyzer() 
        anal_dict= analysis.polarity_scores(tweet)
        if anal_dict['compound']>=0.1:
            return "Positive"
        elif anal_dict['compound']<=-0.1:
            return "Negative"
        else:
            return "Neutral"
    def tweets_to_df(self,tweets):
        df=pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['text'])
        return df

def plot_wordcloud(data):
    wc = WordCloud(background_color='white', width=600, height=500)
    wc.generate(data)
    return wc.to_image()

def open_browser():
	webbrowser.open_new("http://localhost:{}".format(port))

                                                                                       
if __name__ =="__main__":
    port = 8050
    twitter_client=TwitterClient()
    tweet_analyzer=TweetAnalysis()
    api=twitter_client.get_twitter_client_api()
    print("^^^^^^^^^^^^^^^^^^^^Welcome to Twitter Sentiment Analysis Project^^^^^^^^^^^^^^^^^^^^")
    x=int(input("Enter '1' to extract tweets based on a keyword/ '2' to extract tweets by a hashtag: "))
    if(x==2):
           tweets=api.user_timeline(screen_name=input("Enter the name of Twitter User:"),count=int(input("Enter the number of Tweets to be extracted: ")))
           df=tweet_analyzer.tweets_to_df(tweets)
           ori_df=df.copy()
           df['sentiment'] = np.array([tweet_analyzer.analyze_sentiment(tweet) for tweet in df['text']])
    else:
          twitter_streamer=TweetStreamer()
          twitter_streamer.stream_tweets(input("Enter the keyword to be searched: "))
          df1=[]
          with open('tweets.txt','r') as tweets_file:
              for line in tweets_file:
                  tweet1=json.loads(line)
                  df1.append(tweet1)
          df=pd.DataFrame(df1, columns=['text'])
          ori_df=df.copy()
          print(df.head())
          df['sentiment'] = np.array([tweet_analyzer.analyze_sentiment(tweet) for tweet in df['text']])
    print(df.head(20))
    
    positive=len(df[df['sentiment']=="Positive"])
    negative=len(df[df['sentiment']=="Negative"])
    neutral=len(df[df['sentiment']=="Neutral"])
    p_p=(positive/(positive+negative+neutral))*100
    n_p=(negative/(positive+negative+neutral))*100
    nn_p=(neutral/(positive+negative+neutral))*100
    df1=df[df["sentiment"]=="Positive"]
    df2=df[df["sentiment"]=="Negative"]
    df3=df[df["sentiment"]=="Neutral"]
    sentences=ori_df['text'].tolist()
    sentences_total=" ".join(sentences)
    sentences_pos=df1["text"].tolist()
    sentences_pos_total=" ".join(sentences_pos)
    sentences_neg=df2["text"].tolist()
    sentences_neg_total=" ".join(sentences_neg)
    
    
    df11=df1.copy()
    df22=df2.copy()
    df33=df3.copy()
    df11['totalwords'] = df11['text'].str.split().str.len()
    df22['totalwords'] = df22['text'].str.split().str.len()
    df33['totalwords'] = df33['text'].str.split().str.len()
    
    ss1=df11['totalwords'].sum()
    ss2=df22['totalwords'].sum()
    ss3=df33['totalwords'].sum()
    
    sum1=int(df11['totalwords'].sum()/len(df11))
    sum2=int(df22['totalwords'].sum()/len(df22))
    sum3=int(df33['totalwords'].sum()/len(df33))
    
    
    
    app= dash.Dash()
    app.layout=html.Div(children=[html.Div(html.H1("Twitter Sentiment Analysis Dashboard"),style={"color":"white","text-align":
                                                                  "center","background-color":"grey"
                                                                  ,"border-style":"ridge",
                                                                  "display":"inline-block",
                                                                  "width":"100%","height":"20%"}),html.Div(dcc.Graph(                            
                                                                  id='pie-chart',                            
                                                                  figure={                                
                                                                  'data': [                          
                                                                  go.Pie(                                        
                                                                  labels=['Positives', 'Negatives', 'Neutrals'],
                                                                  values=[p_p, n_p, nn_p],     
                                                                  hole=0.3
                                                                  )                              
                                                                  ],
                                                                  'layout':{
                                                                  'showlegend':True,
                                                                  'title':'Distribution of Sentiments',
                                                                  'annotations':[
                                                                   'Positive','Negative','Neutral'                              
                                                                   ]
                                                                   }                             
                                                                   }                        
                                                                   ),style={"color":"yellow","text-align":
                                                                  "center","background-color":"blue"
                                                                  ,"border-style":"ridge",
                                                                  "display":"inline-block","vertical-align":"top",
                                                                  "width":"49%"}),html.Div([dash_table.DataTable(
                                                                   id="table",columns=[
                                                                   {'id': c, 'name': c} for c in df.columns],data=df.to_dict('records'),filter_action='native', style_cell_conditional=[
                                                                   {
                                                                   'if': {'column_id': c},
                                                                   'textAlign': 'left'
                                                                   } for c in ['text', 'sentiment']
                                                                   ],style_data_conditional=[
                                                                   {
                                                                   'if': {'row_index': 'odd'},
                                                                   'backgroundColor': 'rgb(248, 248, 248)'
                                                                    }
                                                                    ],
                                                                   style_header={
                                                                  'backgroundColor': 'rgb(230, 230, 230)',
                                                                  'fontWeight': 'bold'
                                                                   },style_table={'height': '450px', 'overflowY': 'auto'},tooltip_data=[
                                                                   {
                                                                   column: {'value': str(value), 'type': 'markdown'}
                                                                   for column, value in row.items()
                                                                   } for row in df.to_dict('rows')
                                                                   ],
                                                                   tooltip_duration=None)],
                                                                   style={"color":"orange","text-align":
                                                                  "center","background-color":"teal"
                                                                  ,"border-style":"ridge",
                                                                  "display":"inline-block","vertical-align":"top",
                                                                  "width":"49%"}),html.Div(html.H2("Word Clouds"),style={"color":"white","text-align":
                                                                  "center","background-color":"grey"
                                                                  ,"border-style":"ridge",
                                                                  "display":"inline-block",
                                                                  "width":"100%","height":"10%"}),html.Div(children=[html.Div(children=[html.H3("Total Word Cloud",style={"color":"white","text-align":
                                                                  "center","background-color":"grey"
                                                                  ,"border-style":"ridge","width":"100%","height":"10%"}),
                                                                   html.Img(id="image_wc",style={"color":"white","text-align":
                                                                  "center","background-color":"grey"
                                                                  ,"border-style":"ridge",
                                                                  "display":"inline-block","border": "7px solid powderblue"})],style={"display":"inline-block"}),html.Div(children=[html.H3("Positive Word Cloud",style={"color":"white","text-align":
                                                                  "center","background-color":"grey"
                                                                  ,"border-style":"ridge","width":"100%","height":"10%"}),
                                                                   html.Img(id="image_wc_pos",style={"color":"white","text-align":
                                                                  "center","background-color":"grey"
                                                                  ,"border-style":"ridge",
                                                                  "display":"inline-block","border": "7px solid powderblue"})],style={"display":"inline-block"}),html.Div(children=[html.H3("Negative Word Cloud",style={"color":"white","text-align":
                                                                  "center","background-color":"grey"
                                                                  ,"border-style":"ridge","width":"100%","height":"10%"}),
                                                                   html.Img(id="image_wc_neg",style={"color":"white","text-align":
                                                                  "center","background-color":"grey"
                                                                  ,"border-style":"ridge",
                                                                  "display":"inline-block","border": "7px solid powderblue"})],style={"display":"inline-block"})],style={"color":"white","background-color":"white"
                                                                  ,"border-style":"ridge",
                                                                  "display":"inline-block",
                                                                  "width":"100%"}),html.Div(html.H2("Statistics"),style={"color":"white","text-align":
                                                                  "center","background-color":"grey"
                                                                  ,"border-style":"ridge",
                                                                  "display":"inline-block",
                                                                  "width":"100%","height":"10%"}),html.Div(children=[html.Div(children=[html.H3("Total Words in Each Category",style={"color":"white","text-align":
                                                                  "center","background-color":"grey"
                                                                  ,"border-style":"ridge","width":"100%","height":"10%"}),dcc.Graph(                            
                                                                  id='pie-chart2',                            
                                                                  figure={                                
                                                                  'data': [                          
                                                                  go.Pie(                                        
                                                                  labels=["Positive",'Negative','Neutral'],
                                                                  values=[ss1,ss2,ss3],     
                                                                  hole=0.3
                                                                  )                              
                                                                  ],
                                                                  'layout':{
                                                                  'showlegend':True,
                                                                  'title':'Total words in each of the sentiments',
                                                                  'annotations':[
                                                                   'Positive','Negative','Neutral'                              
                                                                   ]
                                                                   }                             
                                                                   }                        
                                                                   )]),html.Div(children=[html.H3("Average words per tweet",style={"color":"white","text-align":
                                                                  "center","background-color":"grey"
                                                                  ,"border-style":"ridge","width":"100%","height":"10%"}),dcc.Graph(
                                                                   id='example-graph',
                                                                   figure={
                                                                   'data': [
                                                                   {'x': [1], 'y': [sum1], 'type': 'bar', 'name': 'Positive'},
                                                                   {'x': [1], 'y': [sum2], 'type': 'bar', 'name': 'Negative'},
                                                                   {'x': [1], 'y': [sum3], 'type': 'bar', 'name': 'Neutral'},
                                                                   ],
                                                                  'layout': {
                                                                  'title': 'Average word count in tweets'
                                                                   }
                                                                   }
                                                                   )])],style={"color":"white","background-color":"white"
                                                                  ,"border-style":"ridge",
                                                                  "display":"inline-block",
                                                                  "width":"100%"}),html.Div(html.H2("Made By: Tushar Ojha"),style={"color":"white","text-align":
                                                                  "center","background-color":"grey"
                                                                  ,"border-style":"ridge",
                                                                  "display":"inline-block",
                                                                  "width":"100%","height":"10%"})])
    @app.callback(Output('image_wc', 'src'), [Input('image_wc', 'id')])
    def make_image(b):
        img = BytesIO()
        plot_wordcloud(data=sentences_total).save(img, format='PNG')
        return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())
    
    @app.callback(Output('image_wc_pos', 'src'), [Input('image_wc_pos', 'id')])
    def make_image_pos(b):
        img = BytesIO()
        plot_wordcloud(data= sentences_pos_total).save(img, format='PNG')
        return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())
    
    @app.callback(Output('image_wc_neg', 'src'), [Input('image_wc_neg', 'id')])
    def make_image_neg(b):
        img = BytesIO()
        plot_wordcloud(data= sentences_neg_total).save(img, format='PNG')
        return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())
    
    Timer(1, open_browser).start();
    app.run_server(port=port)