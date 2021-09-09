import tweepy
# auth
consumer_key= "Aab7LOGZyppaFPNDQCdZ44yK7"
consumer_secret = "CYdlRUzGTAPVrQdaa0XfuoTeDNyo1EHb2AuDsoevmlwWTYUkFM"
access_token = "2182508549-jVF957pWQp2znpsd0kw54soL7TZkFwrJsQTusNV"
access_token_secret= "HW9mwAvbO4KnwgZmk7C4LZa2mKqvTorHPokLeSHRF0FKE"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret )
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
