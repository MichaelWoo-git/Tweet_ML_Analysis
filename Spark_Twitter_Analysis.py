# Imports
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from google_drive_downloader import GoogleDriveDownloader as gdd
from pyspark.sql.functions import col
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression
import nltk
from nltk.corpus import stopwords
import string
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

gdd.download_file_from_google_drive(file_id='0B04GJPshIjmPRnZManQwWEdTZjg',
                                    dest_path='/Users/mwoo/Downloads/trainingandtestdata.zip',
                                    unzip=True)
spark = SparkSession.builder.appName('data_processing').getOrCreate()
training_data = spark.read.csv("/Users/mwoo/Downloads/training.1600000.processed.noemoticon.csv", header=False)
training_data = training_data.toDF("target", 'id', 'date', 'query', 'user_name', 'text')

df = training_data.select('text', 'target')

nltk.download('stopwords')
sp = set(string.punctuation)
stop_words = set(stopwords.words('english'))
extra_words = {"http", "https", "amp", "rt", "t", "c", "the"}
for i in extra_words:
    stop_words.add(i)
stop_words = list(stop_words)
regexTokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(stop_words)
countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)
label_stringIdx = StringIndexer(inputCol="target", outputCol="label")
pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, label_stringIdx])
# Fit the pipeline to training documents
pipelineFit = pipeline.fit(df)
dataset = pipelineFit.transform(df)

(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testData.count()))

# trainingData = dataset
# print("Training Dataset Count: " + str(trainingData.count()))
# print("Test Dataset Count: " + str(testData.count()))
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0, labelCol="label", featuresCol="features")
lrModel = lr.fit(trainingData)
predictions = lrModel.transform(testData)
predictions.filter(predictions['prediction'] == 0).select("text", "probability", "label", "prediction").orderBy(
  "probability", ascending=False).show(n=10, truncate=30)
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
print(evaluator.evaluate(predictions))

import tweepy
from tweepy import OAuthHandler
from tweepy import Stream

ACCESS_TOKEN = "1458842253779161088-QFeO6udaAdHR4VARxaDza1w4LUlooE"
ACCESS_TOKEN_SECRET = "tC7IJDbl5T97Zvu3kE8sdGnmZWC2qxOrkdOv90YkdzIVO"
API_KEY = "KLP5ct26qaVo0KjAgP8O4j4y5"
API_KEY_SECRET = "AbxH3913WIPG0FHIwvVRomul92RWvuOdxRo2ecXR6H0Qgibo29"
auth = tweepy.OAuthHandler(API_KEY, API_KEY_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)
tweet_list = list()


# Subclass Stream to print IDs of Tweets received
class IDPrinter(tweepy.Stream):

    def on_status(self, status):
        tweet_list.append(status.text)
        # print(tweet_list)
        print(status.text)
        if len(tweet_list) == 10:
            Stream.disconnect(self)


# Initialize instance of the subclass
printer = IDPrinter(
    API_KEY, API_KEY_SECRET,
    ACCESS_TOKEN, ACCESS_TOKEN_SECRET
)

# Filter realtime Tweets by keyword
# printer.filter(track=["Spiderman"])
printer.sample(languages=['en'])
# New Pipeline
target = ['null' for x in tweet_list]
print(target)
df_2 = pd.DataFrame({'text':np.array(tweet_list),'target':np.array(target)})
# df_2.columns = ['text','target']
print(df_2.head())
df_2 = spark.createDataFrame(df_2)

#pipeline_1 = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors,label_stringIdx])
# Fit the pipeline to training documents
dataset_1 = pipelineFit.transform(df_2)
dataset_1.show(5)
#dataset_1 = dataset_1.select('text', 'features')

predictions = lrModel.transform(dataset_1)
predictions.show()