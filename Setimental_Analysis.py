import tweepy

# Replace the placeholders with your own keys and tokens
consumer_key = '5XndDoKdwVM7VKNExkwMzPQAS'
consumer_secret = 'ruF4RY4ntg1WQ7X5ksEAgZjZS205AhsGmePu8cLYtK20Gx6rpv'
access_token = '1844075087940685824-B5xYCXoTeXlzwkbNnVFsXkRLgDrIqC'
access_token_secret = '5AuyfBly7Xdq360v0rwJ3zzc6kDpC87u2g1IyyAe7BBxf '

# Set up authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Initialize Tweepy API object
api = tweepy.API(auth)

# Check if authentication was successful
try:
    api.verify_credentials()
    print("Authentication successful")
except tweepy.TweepError as e:
    print(f"Authentication failed: {e}")

# Fetch the most recent tweets from your timeline
tweets = api.home_timeline(count=5)
for tweet in tweets:
    print(f"{tweet.user.name} said: {tweet.text}")


# Fetch a specific user's most recent tweets
username = 'twitter'  # Replace with the target user's username
tweets = api.user_timeline(screen_name=username, count=5)
for tweet in tweets:
    print(f"{tweet.user.name} said: {tweet.text}")


# Search for tweets containing a specific hashtag or keyword
keyword = 'Python'
tweets = api.search(q=keyword, count=5)
for tweet in tweets:
    print(f"{tweet.user.name} said: {tweet.text}")


# Get data for a specific tweet using its ID
tweet_id = 'tweet-id'  # Replace with the tweet ID
tweet = api.get_status(tweet_id)
print(f"Tweet by {tweet.user.name}: {tweet.text}")

import time

# Example of handling rate limit errors
try:
    tweets = api.home_timeline(count=10)
    for tweet in tweets:
        print(tweet.text)
except tweepy.RateLimitError:
    print("Rate limit reached. Sleeping for 15 minutes...")
    time.sleep(15 * 60)  # Sleep for 15 minutes



with open('tweets.txt', 'w') as f:
    for tweet in tweets:
        f.write(f"{tweet.user.name}: {tweet.text}\n")
