import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer



class TwitterClient(object):
    '''
    Generic Twitter Class for sentiment analysis.
    '''

    def __init__(self):
        '''
        Class constructor or initialization method.
        '''

        consumer_key = '5VHAzWDJL1UNKFfuHAXbpnDn7'
        consumer_secret = 'iK2eWCaCnu6tZExDbepPpCOWI9gDcPb0c4UBz9IDbVWD9ga62h'
        access_token = '889841328044949504-zEMV4dZc9AMDZ6FoCvzg1aAdPqJULDx'
        access_token_secret = '842vFy31S2l4IpU3dMnsPr5n7SoK2PL5z3mmIzsulRWUw'

        try:

            self.auth = OAuthHandler(consumer_key, consumer_secret)

            self.auth.set_access_token(access_token, access_token_secret)

            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")

    def clean_tweet(self, tweet):
        '''
        Utility function to clean tweet text by removing links, special characters
        using simple regex statements.
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def get_tweet_sentiment(self, tweet):
        '''
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        '''

        analysis = TextBlob(self.clean_tweet(tweet))

        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'

    def get_tweet_sentiment1(self, tweet):
        sid = SentimentIntensityAnalyzer()

        # Generate sentiment scores
        sentiment_scores =sid.polarity_scores(tweet)

        if sentiment_scores['compound'] > 0:
            return 'positive'
        elif sentiment_scores['compound'] < 0:
            return 'negetive'
        else:
            return 'neutral'


    def get_tweets(self, query, count=10):
        '''
        Main function to fetch tweets and parse them.
        '''

        tweets = []

        try:

            fetched_tweets = self.api.search(q=query, count=count)

            for tweet in fetched_tweets:

                parsed_tweet = {}

                parsed_tweet['text'] = tweet.text

                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)
                parsed_tweet['sentiment1'] = self.get_tweet_sentiment1(tweet.text)
                if tweet.retweet_count > 0:

                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)

            return tweets

        except tweepy.TweepError as e:
            print("Error : " + str(e))



# def positive_tweets(value):
#     api = TwitterClient()
#     tweets = api.get_tweets(query=value, count=5000)
#     ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
#     return ptweets
# def negetive_tweets(value):
#     api = TwitterClient()
#     tweets = api.get_tweets(query=value, count=5000)
#     ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
#     return ntweets
# def percents(value):
#     api = TwitterClient()
#     value = input("Please enter a string:\n")
#     tweets = api.get_tweets(query=value, count=5000)
#     ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
#     ptweets1 = [tweet for tweet in tweets if tweet['sentiment1'] == 'positive']
#
#     ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
#     ntweets1 = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
#
#     pper = 100 * (len(ptweets) / len(tweets))
#     pper1=100 * (len(ptweets1) / len(tweets))
#     nnper = 100 * (len(ntweets) / len(tweets))
#     nnper1 = 100 * (len(ntweets1) / len(tweets))
#     nuper=100 * ((len(tweets) - len(ntweets) - len(ptweets)) / len(tweets))
#     nuper1=100 * ((len(tweets) - len(ntweets1) - len(ptweets1)) / len(tweets))
#
#     return {pper,pper1,nnper,nnper1,nuper,nuper1}



    # print("Negative tweets percentage: {} %".format(100 * len(ntweets) / len(tweets)))
    #
    # print("Neutral tweets percentage: {} % ".format(100 * (len(tweets) - len(ntweets) - len(ptweets)) / len(tweets)))

def main(value):

    api = TwitterClient()
    tweets = api.get_tweets(query=value, count=5000)
    ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
    ptweets1 = [tweet for tweet in tweets if tweet['sentiment1'] == 'positive']



    pper1=("Positive tweets percentage: {} %".format(100 * len(ptweets) / len(tweets)))

    ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
    ntweets1 = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']

    # print("Negative tweets percentage: {} %".format(100 * len(ntweets) / len(tweets)))
    #
    # print("Neutral tweets percentage: {} % ".format(100 * (len(tweets) - len(ntweets) - len(ptweets)) / len(tweets)))
    #
    # print("\n\nPositive tweets:")
    # for tweet in ptweets[:10]:
    #     print(tweet['text'])
    #
    # print("\n\nNegative tweets:")
    # for tweet in ntweets[:10]:
    #     print(tweet['text'])
    #
    #
    #
    #
    # print('=================================NTLK Sentiment==================================================')
    # print("Positive tweets percentage: {} %".format(100 * len(ptweets1) / len(tweets)))
    # print("Negative tweets percentage: {} %".format(100 * len(ntweets1) / len(tweets)))
    # print("Neutral tweets percentage: {} % ".format(100 * (len(tweets) - len(ntweets1) - len(ptweets1)) / len(tweets)))
    # print("\n\nPositive tweets:")
    # for tweet in ptweets1[:10]:
    #     print(tweet['text'])
    #
    # print("\n\nNegative tweets:")
    # for tweet in ntweets1[:10]:
    #     print(tweet['text'])

    return (ptweets,ntweets)

def per(value):
    api = TwitterClient()
    tweets = api.get_tweets(query=value, count=5000)
    ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
    ptweets1 = [tweet for tweet in tweets if tweet['sentiment1'] == 'positive']

    pper1 = ("Positive tweets percentage: {} %".format(100 * len(ptweets) / len(tweets)))

    ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
    ntweets1 = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
    p1=(100 * len(ptweets) / len(tweets))
    p2=(100 * len(ptweets1) / len(tweets))
    n1=(100 * len(ntweets) / len(tweets))
    n2 = (100 * len(ntweets1) / len(tweets))
    nu1=(100 * (len(tweets) - len(ntweets) - len(ptweets)) / len(tweets))
    nu2=(100 * (len(tweets) - len(ntweets1) - len(ptweets1)) / len(tweets))
    fin={0:p1,
         1:p2,
         2:n1,
         3:n2,
         4:nu1,
         5:nu2
         }
    return fin




