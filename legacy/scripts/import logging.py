import logging
import pandas as pd
import boto3
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key
from decimal import Decimal

# Constants
TABLE_NAME = 'twitter-analytics-v2'
RATING_FILE_PATH = 'database/ratingv2.csv'
COLUMN_TYPES = {
    'item_id': 'str',
    'user_who_published': 'str',
    'user_id': 'Int32',
    'ranking': 'Int32',
    'rating': 'Int32',
    'algorithm': 'str',
    'date': 'str'
}

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize resources
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(TABLE_NAME)


def get_tweets_from_user(user_id):
    """Get tweets published by a user."""
    try:
        response = table.query(
            KeyConditionExpression=Key('PK').eq(f'Tweet#AuthorId#{user_id}'))
    except ClientError as e:
        logging.error(f"Failed to get tweets from user {user_id}: {e}")
    else:
        return response['Items']


def get_tweet(user_id, tweet_id):
    """Get a specific tweet published by a user."""
    try:
        response = table.get_item(Key={'PK': f'Tweet#AuthorId#{user_id}', 'SK': f'TweetId#{tweet_id}'})
    except ClientError as e:
        logging.error(f"Failed to get tweet {tweet_id} from user {user_id}: {e}")
    else:
        return response.get('Item', {})


def read_recommendations(file_path):
    """Read recommendations from a CSV file."""
    return pd.read_csv(file_path, dtype=COLUMN_TYPES)


def enhance_recommendations(recommendations):
    """Enhance recommendations with additional tweet data."""
    unique_items = recommendations[['item_id', 'user_who_published']].drop_duplicates()
    for index, row in unique_items.iterrows():
        tweet = get_tweet(str(row['user_who_published']), str(row['item_id']))
        if tweet:
            unique_items.at[index, 'score'] = Decimal(tweet.get('SocialCapitalScore', 0))
            unique_items.at[index, 'text'] = tweet.get('Text', '')
            unique_items.at[index, 'tokens'] = tweet.get('Tokens', [])
            unique_items.at[index, 'like_count'] = Decimal(tweet.get('LikeCount', 0))
            unique_items.at[index, 'retweet_count'] = Decimal(tweet.get('RetweetCount', 0))
            unique_items.at[index, 'reply_count'] = Decimal(tweet.get('ReplyCount', 0))
            unique_items.at[index, 'quote_count'] = Decimal(tweet.get('QuoteCount', 0))
    return unique_items

def main():
    recommendations_df = read_recommendations(RATING_FILE_PATH)
    logging.info(recommendations_df.head())
    enhanced_recommendations = enhance_recommendations(recommendations_df)
    # Process or output enhanced recommendations as needed


if __name__ == "__main__":
    main()
