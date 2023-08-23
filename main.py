import pandas as pd
import boto3
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key
from decimal import Decimal

# Set DB and Tale names
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('twitter-analytics-v2')

# get tweets published by user
def get_tweets_from_user(userId):
        try:
            response = table.query(
                KeyConditionExpression=(Key('PK').eq(f'Tweet#AuthorId#{userId}')))
        except ClientError as err:
           print(err.response['Error']['Message'])
        else:
            return response['Items']

# get a specific tweet published by user
def get_tweet(userId, tweetId):
        try:
            response = table.get_item(Key={'PK': f'Tweet#AuthorId#{userId}', 'SK': f'TweetId#{tweetId}'})
        except ClientError as err:
           print(err.response['Error']['Message'])
        else:
            return response['Item']
        
column_types = {
    'item_id': 'str',
    'user_who_published': 'str',
    'user_id': 'Int32',
    'ranking': 'Int32',
    'rating': 'Int32',
    'algorithm': 'str',
    'date': 'str'
}

# Read a ratings DB
recommendationsDf = pd.read_csv('database/ratingv2.csv', dtype=column_types) 

# Print a sample of recommendations        
print(recommendationsDf.head())

# Get all recommended tweet and drop duplicates
unique_items = recommendationsDf[['item_id', 'user_who_published']].drop_duplicates()
for index, row  in unique_items.iterrows():
     who_published = str(row['user_who_published'])
     tweet_id = str(row['item_id'])

     tweet = get_tweet(who_published, tweet_id)
     unique_items.at[index, 'score'] = Decimal(tweet['SocialCapitalScore'])
    
# create a dataset from the new approach
newRecommendations = pd.DataFrame()
for index, row in recommendationsDf.iterrows():
     user_id = row['user_id']
     who_published = str(row['user_who_published'])
     tweet_id = str(row['item_id'])
     algorithm = row['algorithm']

     rating = recommendationsDf[(recommendationsDf['item_id'] == tweet_id) & (recommendationsDf['algorithm'] == algorithm) & (recommendationsDf['user_id'] == user_id)]['rating'].values[0]

     score = unique_items[unique_items['item_id'] == tweet_id]['score'].values[0]
     newRecommendations.at[index, 'item_id'] = tweet_id
     newRecommendations.at[index, 'user_who_published'] = who_published
     newRecommendations.at[index, 'user_id'] = user_id
     newRecommendations.at[index, 'ranking'] = 0
     newRecommendations.at[index, 'rating'] = rating
     newRecommendations.at[index, 'score'] = score
     newRecommendations.at[index, 'algorithm'] = f'{algorithm}-SCSA_PLUS'
     newRecommendations.at[index, 'date'] = "2023-08-19"

ranking = 1
sortRecommendationsByUserAndAlgorithm = newRecommendations.sort_values(by=['user_id','algorithm', 'score'], ascending=False)

for index, row in sortRecommendationsByUserAndAlgorithm.iterrows():
    sortRecommendationsByUserAndAlgorithm.at[index, 'ranking'] = ranking
    if(ranking == 10):
         ranking = 1
    else: 
         ranking = ranking + 1

print(sortRecommendationsByUserAndAlgorithm.head())
sortRecommendationsByUserAndAlgorithm.drop('score', axis=1, inplace=True)

combined_df = pd.concat([recommendationsDf, sortRecommendationsByUserAndAlgorithm], ignore_index=True)
combined_df.to_csv('output.csv', index=False)
print()
# Print the fetched item
# items = get_tweets_from_id(715177919438577664, 1317842770816602114)
#print(items)

