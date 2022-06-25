import glob
import json
import pandas as pd
from tweepy import OAuthHandler
from tweepy import API
from tweepy import Stream,StreamListener
import numpy as np


# assign the values accordingly
consumer_key = '5VHAzWDJL1UNKFfuHAXbpnDn7'
consumer_secret = 'iK2eWCaCnu6tZExDbepPpCOWI9gDcPb0c4UBz9IDbVWD9ga62h'
access_token = '889841328044949504-zEMV4dZc9AMDZ6FoCvzg1aAdPqJULDx'
access_token_secret = '842vFy31S2l4IpU3dMnsPr5n7SoK2PL5z3mmIzsulRWUw'



# Consumer key authentication(consumer_key,consumer_secret can be collected from our twitter developer profile)
auth = OAuthHandler(consumer_key, consumer_secret)

# Access key authentication(access_token,access_token_secret can be collected from our twitter developer profile)
auth.set_access_token(access_token, access_token_secret)

# Set up the API with the authentication handler
api = API(auth)

## Set up words to track
keywords_to_track = ['#javascript','#python']

# Instantiate the SListener object
listen = StreamListener(api)

# Instantiate the Stream object
stream = Stream(auth, listen)

# Begin collecting data
stream.filter(track = keywords_to_track)


tweets = []
files = list(glob.iglob('/content/tweets.json'))
for f in files:
    fh = open(f, 'r', encoding='utf-8')
    tweets_json = fh.read().split("\n")

    ## remove empty lines
    tweets_json = list(filter(len, tweets_json))

    ## parse each tweet
    for tweet in tweets_json:
        tweet_obj = json.loads(tweet)

        # Store the user screen name in 'user-screen_name'
        tweet_obj['user-screen_name'] = tweet_obj['user']['screen_name']

        # Check if this is a 140+ character tweet
        if 'extended_tweet' in tweet_obj:
            # Store the extended tweet text in 'extended_tweet-full_text'
            tweet_obj['extended_tweet-full_text'] = tweet_obj['extended_tweet']['full_text']

        if 'retweeted_status' in tweet_obj:
            # Store the retweet user screen name in 'retweeted_status-user-screen_name'
            tweet_obj['retweeted_status-user-screen_name'] = tweet_obj['retweeted_status']['user']['screen_name']

            # Store the retweet text in 'retweeted_status-text'
            tweet_obj['retweeted_status-text'] = tweet_obj['retweeted_status']['text']

        if 'quoted_status' in tweet_obj:
            # Store the retweet user screen name in 'retweeted_status-user-screen_name'
            tweet_obj['quoted_status-user-screen_name'] = tweet_obj['quoted_status']['user']['screen_name']

            # Store the retweet text in 'retweeted_status-text'
            tweet_obj['quoted_status-text'] = tweet_obj['quoted_status']['text']

        tweets.append(tweet_obj)

df_tweet = pd.DataFrame(tweets)


def check_word_in_tweet(word, data):
    """Checks if a word is in a Twitter dataset's text.
    Checks text and extended tweet (140+ character tweets) for tweets,
    retweets and quoted tweets.
    Returns a logical pandas Series.
    """
    contains_column = data['text'].str.contains(word, case=False)
    contains_column |= data['extended_tweet-full_text'].str.contains(word, case=False)
    contains_column |= data['quoted_status-text'].str.contains(word, case=False)
    contains_column |= data['retweeted_status-text'].str.contains(word, case=False)
    return contains_column


# Find mentions of #python in all text fields
python = check_word_in_tweet('python', df_tweet)
# Find mentions of #javascript in all text fields
js = check_word_in_tweet('javascript', df_tweet)

# Print proportion of tweets mentioning #python
print("Proportion of #python tweets:", np.sum(python) / df_tweet.shape[0])

# Print proportion of tweets mentioning #rstats
print("Proportion of #javascript tweets:", np.sum(js) / df_tweet.shape[0])


# Print created_at to see the original format of datetime in Twitter data
print(df_tweet['created_at'].head())

# Convert the created_at column to np.datetime object
df_tweet['created_at'] = pd.to_datetime(df_tweet['created_at'])

# Print created_at to see new format
print(df_tweet['created_at'].head())

# Set the index of df_tweet to created_at
df_tweet = df_tweet.set_index('created_at')

# Create a python column
df_tweet['python'] = check_word_in_tweet('python', df_tweet)

# Create an js column
df_tweet['js'] = check_word_in_tweet('javascript', df_tweet)

import matplotlib.pyplot as plt
# Average of python column by day
mean_python = df_tweet['python'].resample('1 min').mean()

# Average of js column by day
mean_js = df_tweet['js'].resample('1 min').mean()

# Plot mean python/js by day
plt.plot(mean_python.index.minute, mean_python, color = 'green')
plt.plot(mean_js.index.minute, mean_js, color = 'blue')

# Add labels and show
plt.xlabel('Minute'); plt.ylabel('Frequency')
plt.title('Language mentions over time')
plt.legend(('#python', '#js'))
plt.show()