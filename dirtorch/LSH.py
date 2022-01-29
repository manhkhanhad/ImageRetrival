import argparse
import sys
from query import *
from utils.common import tonumpy, matmul, pool
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
    def __init__(self, train_feature, bucketLength = 0.0002, numHashTables = 3):
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
        return resultList
if __name__ == "__main__":
    net = load_model(os.path.join(os.environ['DIR_ROOT'],"checkpoints/Resnet-101-AP-GeM.pt"),False)
    #Load data
    dataset = ImageListRelevants(os.environ["DB_ROOT"]+"/oxford5k/gnd_oxford5k.pkl", os.environ["DB_ROOT"] + "/oxford5k", "jpg" )
    dataLoader = get_loader(dataset, trf_chain="", preprocess=net.preprocess, iscuda=True,output=['img'], batch_size=1, threads=1, shuffle=False)
    #Load database feature
    bdescs = np.load(os.path.join(os.environ['DB_ROOT'] + "/oxford5k", 'feats.bdescs.npy'))
    lshash = LSH(bdescs)
    
    query = cv2.imread(os.environ['DIR_ROOT']+'/query.jpg')
    transforms_img = transforms.Compose([transforms.ToTensor()])
    query_img = transforms_img(query).unsqueeze(0)
    query_img = extractQueryFeature(query_img,net)
    
    res = lshash.predict(query_img)
    
    inx_ft = []
    for i in res:
      inx_ft.append(bdescs[i])
    inx_ft = tonumpy(np.array(inx_ft))
    query_feature = tonumpy(np.array(query_img))
    scores = matmul(query_feature, inx_ft)
    ranklist = sorted(range(len(scores)), key=lambda i: scores[i])[-10:]
    ranklist = ranklist[::-1] #reverse
    print(len(ranklist))
    with open(os.environ['DIR_ROOT']+'test.txt', 'w') as f:
        for item in ranklist:
            f.write("%s\n" % res[item])