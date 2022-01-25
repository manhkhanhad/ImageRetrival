
from dirtorch.query import *
import os
import cv2
from PIL import Image
from flask import Flask
#from flask_cors import CORS, cross_origin
from flask import request

import numpy as np
import cv2
import base64
from io import BytesIO
import time
from flask import jsonify

#app = Flask(__name__, static_folder="static")

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
        #img = cv2.imread(img_path)
        img = Image.open(img_path).convert('RGB')
        relevant_image_path.append(img_path)
        relevant_image.append(PILtobase64(img))
        
        print(img_path)
        #cv2_imshow(resizeImg(img,0.20))
    
    result = {
                'query_time': query_time,
                'top_k_score': top_k_score,
                'relevant_image_path': relevant_image_path,
                'relevant_image': relevant_image
                }
    print(result["query_time"])
    print(result["top_k_score"])
    print(result["relevant_image_path"])
    for img in result["relevant_image"]:
        print(len(img))
    
    return result
    
def base64toPIL(img):
    im_bytes = base64.b64decode(img)
    im_file = BytesIO(im_bytes)
    return Image.open(im_file)

def PILtobase64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str


# @app.route('/query', methods=['POST'])
# def getQueryFromClient():
#     """
#     Parameter:
#         Input: 
#         {
#             'query': (base64) query image
#             'top_k': (int) number of relevant image to response
#         }
    
#         Output: 
#         {
#             query_time: (float) query time in second
#             top_k_score: (list of float) similarity score of relevant image
#             relevant_image_path: (list of string) path of relevant image
#             relevant_image: (list of base64) relevant images
#         }
#     """
    
#     #Read Image form client and convert to PIL Image
#     query_img = request.form.get('query')
#     query_img = base64toPIL(query_img)
#     query_img = transforms_img(query_img).unsqueeze(0)
    
#     #Query
#     result = query(query_img,net,10)
    
#     #Return result to Client
#     response = {
#         'status': 200,
#         'message': 'OK',
#         'result': result
#     }
    
#     response = jsonify(response)
#     return(response)
    
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
    
    
    #Read query image
    query_path = os.path.join(os.environ['DIR_ROOT'],"query.jpg")
    query_img = Image.open(query_path).convert('RGB')
    query_img = transforms_img(query_img).unsqueeze(0)
    query(query_img,net,10)
    
    #app.run(host='0.0.0.0', port='6868')