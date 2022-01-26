
from dirtorch.query import *
from dirtorch.LSH import *
import os
import cv2
from PIL import Image
from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request
import re

import numpy as np
import cv2
import base64
from io import BytesIO
import time
from flask import jsonify
from flask import send_file, send_from_directory, safe_join, abort
import random

app = Flask(__name__, static_folder="static")
cors = CORS(app, resources={r"/foo": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'


def query(query_img, net, topk):
    #query_img = cv2.imread(query_path)
    #cv2_imshow(resizeImg(query_img,0.20))
    start = time.time()
    query_feature = extractQueryFeature(query_img,net)
    query_feature = tonumpy(query_feature)
    scores = matmul(query_feature, bdescs)
    
    ranklist = sorted(range(len(scores)), key=lambda i: scores[i])[-topk:]
    ranklist = ranklist[::-1] #reverse
    query_time = time.time() - start
    
    
    top_k_score = sorted(scores)[-1:-topk-1:-1]
    
    relevant_image_path = []
    relevant_image = []
    for i in ranklist:
        img_path = dataset.get_filename(i)
        img = Image.open(img_path).convert('RGB')
        image_name = img_path.split('/')[-1]
        relevant_image_path.append(image_name)
    
    result = {
                'query_time': str(query_time),
                'top_k_score': str(top_k_score),
                'relevant_image_name': relevant_image_path
                }
    print(result["query_time"])
    print(result["top_k_score"])
    print(result["relevant_image_name"])
    
    return result

def query_hash(query_img, net, topk, LSH):
    #query_img = cv2.imread(query_path)
    #cv2_imshow(resizeImg(query_img,0.20))
    start = time.time()
    query_feature = extractQueryFeature(query_img,net)
    query_feature = tonumpy(query_feature)
    
    ranklist = LSH.predict(query = query_feature, top = topk)
    ranklist = ranklist[::-1] #reverse
    query_time = time.time() - start
    
    
    #top_k_score = sorted(scores)[-1:-topk-1:-1]
    
    relevant_image_path = []
    relevant_image = []
    for i in ranklist:
        img_path = dataset.get_filename(i)
        img = Image.open(img_path).convert('RGB')
        image_name = img_path.split('/')[-1]
        relevant_image_path.append(image_name)
    
    result = {
                'query_time': str(query_time),
                #'top_k_score': str(top_k_score),
                'relevant_image_name': relevant_image_path
                }
    print(result["query_time"])
    #print(result["top_k_score"])
    print(result["relevant_image_name"])
    
    return result
    
def base64toPIL(img):
#     im_bytes = base64.b64decode(img)
#     im_file = BytesIO(im_bytes)
#     return Image.open(im_file)
    img = re.sub('^data:image/.+;base64,', '', img)
    img = Image.open(BytesIO(base64.b64decode(img)))
    return img

def PILtobase64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str

@app.route('/', methods=['GET'])
def test():
    return "server is running"

@app.route("/get-image/<image_name>",methods=['GET'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def getImage(image_name):
    try:
        response = send_from_directory(os.path.join(os.environ['DB_ROOT'] + "/oxford5k/jpg"), path=image_name, as_attachment=True)
        return response
    except FileNotFoundError:
        abort(404)

@app.route("/get-suggest-query",methods=['POST'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def suggestQuery():
    """
    Parameter:
        Input:
        {
        'category': (str) image category
        }
        
        Output:
        {
        'result': (list of str) name of images
        }
    
    Available Categories
    ['christ_church', 'trinity', 'magdalen', 'oxford', 'balliol', 'hertford', 'radcliffe_camera', 'new', 'ashmolean', 'all_souls', 'oriel', 'worcester', 'bodleian', 'cornmarket', 'pitt_rivers', 'keble', 'jesus']
    """
    
    category = request.form.get('category')
    
    if category in image_dict.keys():
        result = image_dict[category]
        random.shuffle(result)
    else:
        result = random.sample(os.listdir(os.path.join(os.environ['DB_ROOT'] + "/oxford5k/jpg")), 20)
        
    response = {
        'status': 200,
        'message': 'OK',
        'result': result
    }
    return response

@app.route('/query', methods=['POST'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def getQueryFromClient():
    """
    Parameter:
        Input: 
        {
            'query': (base64) query image
            'top_k': (int) number of relevant image to response
        }
    
        Output: 
        {
            query_time: (float) query time in second
            top_k_score: (list of float) similarity score of relevant images
            relevant_image_name: (list of string) name of relevant images
        }
    """
    
    #Read Image form client and convert to PIL Image
    query_img = request.form.get('query')
    query_img = base64toPIL(query_img)
    query_img = transforms_img(query_img).unsqueeze(0)
    
    #Query
    LSH = LSH(bdescs)
    result = query_hash(query_img,net,10,LSH)
    
    #Return result to Client
    response = {
        'status': 200,
        'message': 'OK',
        'result': result,
    }
    
    # add header

    return(response)


if __name__ == "__main__":
    
    #Load Network
    net = load_model(os.path.join(os.environ['DIR_ROOT'],"checkpoints/Resnet-101-AP-GeM.pt"),False)

    #Load data
    dataset = ImageListRelevants(os.environ["DB_ROOT"]+"/oxford5k/gnd_roxford5k.pkl", os.environ["DB_ROOT"] + "/oxford5k", "jpg" )
    dataLoader = get_loader(dataset, trf_chain="", preprocess=net.preprocess, iscuda=True,
                            output=['img'], batch_size=1, threads=1, shuffle=False)
    #Load database feature
    bdescs = np.load(os.path.join(os.environ['DB_ROOT'] + "/oxford5k", 'feats.bdescs.npy'))

    #Create image transform 
    transforms_img = transforms.Compose([transforms.ToTensor()])
    
    #Create Image categories for
    image_dict = {}
    for image_name in os.listdir(os.path.join(os.environ['DB_ROOT'] + "/oxford5k/jpg")):
        category = image_name[:image_name.find('0')-1]
        if category not in image_dict.keys():
            image_dict[category] = [image_name]
        else:
            image_dict[category].append(image_name)
    
    
    #Read query image
#     query_path = os.path.join(os.environ['DIR_ROOT'],"query.jpg")
#     query_img = Image.open(query_path).convert('RGB')
#     query_img = transforms_img(query_img).unsqueeze(0)
#     query(query_img,net,10)
    #hash_query
    #query_path = os.path.join(os.environ['DIR_ROOT'],"query.jpg")
    #query_img = Image.open(query_path).convert('RGB')
    #query_img = transforms_img(query_img).unsqueeze(0)
    #LSH = LSH(bdescs)
    #query_hash(query_img,net,10)
    
    app.run(host='0.0.0.0', port='6868')