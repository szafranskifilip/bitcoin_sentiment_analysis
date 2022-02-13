# Bitcoin sentiment and price analysis. Search for patterns affecting price changes. 

---
### Table of contents:
1. Twitter sentiment analysis and its importance for business analysis
2.  Sentiment analysis as an additional component for trading algorithms
3. Data sourcing
4. Data overview and statistics
5. Neural network classification model
6. Exploratory data analysis and results
7. Summary and next staps


<br />

### 1. Twitter sentiment analysis and its importance for business analysis
 
Sentiment analysis refers to identifying as well as classifying the sentiments that are expressed in the text source. It aims to analyze people's sentiments, attitudes, opinions, emotions, etc. towards elements such as products, individuals, topics ,organizations, and services.


Why Twitter? The problem is to understand your audience, keep on top of what’s being said about your brand – and your competitors – but also discover new trends in the industry. Twitter sentiment analysis allows you to keep track of what's being said about your product or service on social media, and can help you detect angry customers or negative mentions before they escalate.

**Twitter Sentiment Analysis Use Cases**

Twitter sentiment analysis provides many exciting opportunities for businesses. Being able to analyze tweets, and determine the sentiment of each message, adds a new dimension to social media monitoring. The examples of the most popular use cases are the following:

- Social Media Monitoring
- Customer Service
- Market Research
- Political Campaigns

Of course there are many more cases where sentiment analysis can be used to provide valuable feedback. In this project we’re exploring how analyzing the Twitter content can help in finding micro patterns that could be a trigger for a price change. A use case for this analysis could be a trading algorithm.

<br />

### 2. Sentiment analysis as an additional component for trading algorithms

#### *`“With this study, we understood that Bots only depend upon indicators, whereas the volatility induced by Elon Musk tweets or the sentiment due to Chinese government ban on crypto was not embedded. And it requires a human to intervene now and then. “`*


Trading bots mostly depend on the technical analysis and typical trading indicators and patterns. While this can be good for the stock market, it might fail with the cryptocurrencies due to the high volatility.
To improve the accuracy and returns of the cryptocurrency trading algorithms, we could add a sentiment analysis component to track investors’ emotions towards cryptocurrencies. This feature could be another indicator to improve crypto trading. 

<br />

### 3. Data Sourcing

**Twitter Data**

The data used in this project falls into two groups - Twitter content and BTC price data.
The tweets were downloaded directly from Twitter using Twitter API and Python library - tweetpy. The tweet content was also split into 3 csv files that include respectively: 

- tweets data that includes basic tweets information and public metrics:

```
tweet _id, author_id, created_at 
source
public_metrics["retweet_count"]
public_metrics["reply_count"] 
public_metrics["like_count"] 
public_metrics["like_count"]
text
```

- author data

```
author_id, 
username, 
public_metrics["followers_count"]
public_metrics["following_count"]
public_metrics["tweet_count"]
public_metrics["listed_count"]
```
- tweet counts for each day. 

Characteristic of the API query:
- query= *```"(#btc OR #bitcoin OR bitcoin) is:verified -has:media -is:retweet lang:en"```*
- Due to the API limitations, the tweets’ data can only include the last 7 days. 
- tweets were only sourced from the verified users with significant audience
- Tweets were written in english to allow easy analysis
- Tweets don’t include any images or videos to focus only on text content classification

**BTC Price Data**

The bitcoin price data was sourced directly from min-api.cryptocompare.com to obtain hourly price data. 
BTC price data was saved as a csv file and contain information like:

```
{'time': 1642748400,
 'high': 39173.58,
 'low': 38575.8,
 'open': 38697.49,
 'volumefrom': 1758.17,
 'volumeto': 68334552.69,
 'close': 39142.31,
```

Characteristic of the API query:
```
payload = {
    "api_key": apiKey,
    "fsym": "BTC",
    "tsym": "USD",
    "limit": 250
```

The price data includes last 250 hours (last 10 days)


**Importance of data features for sentiment and price analysis**

The most important element of the twitter data is the text content which allows it to perform the sentiment analysis. However, other elements like twitter counts, author id or the amount of followers and retweets may also affect the price to some degree. 

**Limitations**

Major limitation that doesn’t allow for more deep exploration and analysis is the twitter API limit. Without access to the academic account there is no possibility to download the historical data, only the last 7 days. 

The twitter data also doesn;t include other major languages and geo data which may have a significant impact on the results. 


### 4. Data Overview and Statistics

**Tweet data**


![tweet_data](img/tweet_data.png)

![tweet_data](img/author_data.png)

![tweet_data](img/tweet_counts.png)




<br />
<br />


## 5. Neural network Classification Model

**Raw Data Labeling Process**

For the best results twitter text content should be labeled manually, however in this case, due to time constraints, Twitter data was labeled using VADER (Valence Aware Dictionary and sEntiment Reasoner). 

Vader is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. VADER uses a combination of A sentiment lexicon is a list of lexical features (e.g., words) which are generally labeled according to their semantic orientation as either positive or negative. VADER not only tells about the Positivity and Negativity score but also tells us about how positive or negative a sentiment is.

![tweet_data](img/tweet_vader.png)

The Compound score is a metric that calculates the sum of all the lexicon ratings which have been normalized between -1(most extreme negative) and +1 (most extreme positive).
- positive sentiment : (compound score >= 0.05) 
- neutral sentiment : (compound score > -0.05) and (compound score < 0.05) 
- negative sentiment : (compound score <= -0.05)

![tweet_data](img/tweet_labels.png)

**Cleaning and Preprocessing Data**

Keras provides a Tokenizer class that can be fit on the training data, can convert text to sequences consistently by calling the texts_to_sequences() method on the Tokenizer class, and provides access to the dictionary mapping of words to integers in a word_index attribute.

1. Address class weight imbalance by providing class weights with sklearn compute_class_weight function
2. Create and remove typical stopwords and punctuation that occur in English language with nltk.word_tokenize
3. Check tweet length distribution and max value (including stopwords)
4. Vectorize tweets using Tokenizer class from keras.preprocessing.text library by vectorizing a tweets’ text corpus by turning each text into either a sequence of integers
5. Truncate and pad input sequences to be all the same length vectors with keras.preprocessing.sequence.pad_sequences. This function transforms a list of sequences (lists of integers) into a 2D Numpy array of shape (num_samples, num_timesteps). num_timesteps is either the maxlen argument if provided, or the length of the longest sequence in the list.
    - Sequences that are shorter than num_timesteps are padded with value until they are num_timesteps long.
    - Sequences longer than num_timesteps are truncated so that they fit the desired length.

```python
# This class allows to vectorize a text corpus, by turning each text into either a sequence of integers
sequences = t.texts_to_sequences(data['text'])

# Find longest tweet in sequences
def max_tweet():
    for i in range(1, len(sequences)):
        max_length = len(sequences[0])
        if len(sequences[i]) > max_length:
            max_length = len(sequences[i])
    return max_length

max_tweet_len = max_tweet()

#Truncate and pad input sequences to be all the same lenght vectors
padded_data = pad_sequences(sequences, maxlen=max_tweet_len)
padded_data
```


``` 
array([[ 6023,  3194,  6024, ...,     3,     4,  6028],
       [    0,     0,     0, ...,     3,     4,  6029],
       [    0,     0,     0, ...,     3,     4,  6030],
       ...,
       [    0,     0,     0, ...,     3,     4, 15043],
       [    0,     0,     0, ...,     3,     4, 15046],
       [    0,     0,     0, ...,     2,     1, 15049]], dtype=int32)
```

<br />
6. Split data into training and test sets using train_test_split function. Check the shape of data after splitting

```python
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_data, target, test_size = 0.2, random_state = 0)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Size of train and test datasets
print('X_train size:', X_train.shape)
print('y_train size:', y_train.shape)
print('X_test size:', X_test.shape)
print('y_test size:', y_test.shape)
```

```
X_train size: (3812, 45)
y_train size: (3812, 3)
X_test size: (954, 45)
y_test size: (954, 3)
```

<br />



---


