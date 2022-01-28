import argparse
import sys
from os import listdir
from os.path import isfile, join
from typing import Dict, List, Optional, Tuple
import imagehash
from PIL import Image
import cv2
from collections import Counter
import scipy as sp
import numpy as np # Import numpy library 
from skimage.feature import hog # Import Hog model to extract features
from sklearn.metrics import confusion_matrix # Import confusion matrix to evaluate the performance
import pandas as pd
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
from pyspark.conf import SparkConf
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.sql import SparkSession
class LSH():
    def __init__(self, train_feature, bucketLength = 0.008, numHashTables = 18):
        """
        push train_feature to hash 's bucket 
        train_feature: array of images_feature (np ndarray)
        bucketLength: length of bucket
        numHashTables: num of hashtables
        """
        spark = SparkSession.builder \
          .master("local") \
          .appName("Image Retrieval") \
          .config("spark.some.config.option", "some-value") \
          .getOrCreate()
        df = pd.DataFrame(train_feature)
        df['id'] = np.arange(1,len(df)+1,1)
        train = df.values
        Train = map(lambda x: (int(x[-1]),Vectors.dense(x[:-1])), train)
        Train_df = spark.createDataFrame(Train,schema=["id","features"])
        
        brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", 
                                      bucketLength=bucketLength,numHashTables=numHashTables)
        model = brp.fit(Train_df)
        model.transform(Train_df)
        self.model = model 
        self.train = Train_df
    def predict(self, query, top = 10):
        """
        return top relevant images id
        query: feature of query
        """
        key = Vectors.dense(query)
        result = self.model.approxNearestNeighbors(self.train, key, 10000, distCol="EuclideanDistance")
        resultList = result.select("id").rdd.flatMap(lambda x: x).collect()
        return resultList[:top]