# import the module
import tweepy
def trending():
	# assign the values accordingly
	consumer_key = '5VHAzWDJL1UNKFfuHAXbpnDn7'
	consumer_secret = 'iK2eWCaCnu6tZExDbepPpCOWI9gDcPb0c4UBz9IDbVWD9ga62h'
	access_token = '889841328044949504-zEMV4dZc9AMDZ6FoCvzg1aAdPqJULDx'
	access_token_secret = '842vFy31S2l4IpU3dMnsPr5n7SoK2PL5z3mmIzsulRWUw'

# authorization of consumer key and consumer secret
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

# set access to user's access key and access secret
	auth.set_access_token(access_token, access_token_secret)

# calling the api
	api = tweepy.API(auth)

# WOEID of London
	woeid = 44418

# fetching the trends
	trends = api.trends_place(id = woeid)

	return trends




