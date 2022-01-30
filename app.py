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

import kornia as K
import kornia.feature as KF
from kornia.feature import laf_from_center_scale_ori as get_laf
from copy import deepcopy
import pydegensac
from extract_patches.core import extract_patches
from affnet.AFFNetExtractor import extract_descriptors, extract_sift_keypoints_upright, extimate_affine_shape, estimate_orientation, match_snn, ransac_validate
import json

from scipy.spatial import distance as dist

app = Flask(__name__, static_folder="static")
cors = CORS(app, resources={r"/foo": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
KEYPOINT_ROOT= "/home/jovyan/work/ImageMatching/Keypoints"

def AFFNetExtractor(img):
    NFEATS = 2500
    dev = 'cpu'
    open_cv_image = np.array(img)
    kpts = extract_sift_keypoints_upright(open_cv_image, NFEATS)
    As = extimate_affine_shape(kpts, open_cv_image, affnet, dev, ellipse=True)
    ori = estimate_orientation(kpts, open_cv_image, As, orinet, dev)
    kpts_new = [cv2.KeyPoint(x.pt[0], x.pt[1], x.size, ang) for x, ang in zip(kpts,ori)]
    descs = extract_descriptors(kpts_new, open_cv_image, As, hardnet, dev)
    
    return kpts_new, descs, As, open_cv_image

def load_npz(file_path):
    loaded = np.load(file_path)
    pts = []
    for point in loaded["pts"]:
        pts.append(cv2.KeyPoint(point[0], point[1], point[2], point[3]))
    return pts, loaded["descs"], loaded["As"]

def order_points(pts):
	xSorted = pts[np.argsort(pts[:, 0]), :]
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	return np.array([tl, tr, br, bl], dtype="float32")

def draw_bbox(kps1, kps2, As1, As2, tentatives, inl_mask, img1, img2, H_gt, H):
    matchesMask = inl_mask.ravel().tolist()
    tentatives_idxs = np.array([[m.queryIdx, m.trainIdx] for m in tentatives])
    h,w,ch = img1.shape
    pts = np.float32( [[0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, H)
    #Ground truth transformation
    #img2_tr = cv2.polylines(img2,[np.int32(dst)],True,(0,0,255),3, cv2.LINE_AA)
    bbox = np.int32(dst).reshape(4,2)
    negative = (bbox < 0)
    bbox[negative] = 0

    h,w, _ = img2.shape
    

    bbox = order_points(bbox)
    for i in range(4):
        if bbox[i,0] > w:
            bbox[i,0] = w
        if bbox[i,1] > h:
            bbox[i,0] = h
    top = min(bbox[:,1])
    bot = max(bbox[:,1])
    left = min(bbox[:,0])
    right = max(bbox[:,0])
    # x = bbox[:,0]
    # y = bbox[:,1]
    # x.sort()
    # y.sort()

    # top = x[1]
    # bot = x[2]
    # left = y[1]
    # right = y[2]

    return [int(top), int(bot), int(left), int(right)]

def query(query_img, net, topk, getbbox):
    #query_img = cv2.imread(query_path)
    #cv2_imshow(resizeImg(query_img,0.20))
    start = time.time()
    query_feature = extractQueryFeature(query_img,net)
    query_feature = tonumpy(query_feature)
    scores = matmul(query_feature, bdescs)
    
    ranklist = sorted(range(len(scores)), key=lambda i: scores[i])[-topk:]
    ranklist = ranklist[::-1] #reverse
    
    
    
    top_k_score = sorted(scores)[-1:-topk-1:-1]
    
    relevant_image_path = []
    relevant_image = []
    
    for i in ranklist:
        img_path = dataset.get_filename(i)
        img = Image.open(img_path).convert('RGB')
        image_name = img_path.split('/')[-1]
        relevant_image_path.append(image_name)

    bbox_dict = {}   
    #Spaital Matching
    if getbbox == True:
        bbox = []
        num_match_keypoint = []
        q_kpts, q_descs, q_As, query_im_cv = AFFNetExtractor(img)
        for img_name in relevant_image_path:
            relevant_img = cv2.cvtColor(cv2.imread(os.path.join(os.environ['DB_ROOT'] + "/oxford5k/jpg",img_name)), cv2.COLOR_BGR2RGB)
            kpts, descs, As = load_npz(os.path.join(KEYPOINT_ROOT, img_name[:-4] + ".npz"))
            tentatives = match_snn(q_descs, descs, 0.85)
            if len(tentatives) >= 4:
                H, inl_mask = ransac_validate(tentatives, q_kpts, kpts, 2.0)  #
                box = draw_bbox(q_kpts, kpts, q_As, As, tentatives, inl_mask, query_im_cv, relevant_img, Hgt, H)
                num_match_keypoint.append(len(tentatives))
            else:
                box = [0,0,0,0]
                num_match_keypoint.append(0)
                
            bbox.append(box)
            
        rerank = np.argsort(num_match_keypoint)[::-1]
        
        top_k_score = [top_k_score[i] for i in rerank]
        relevant_image_path = [relevant_image_path[i] for i in rerank]
        bbox = [bbox[i] for i in rerank]

        for r,b in enumerate(bbox):
            bbox_dict[r+1] = b
        print(bbox_dict)      
        
    query_time = time.time() - start
    
    result = {
                'query_time': round(query_time, 2),
                'top_k_score': str(top_k_score),
                'relevant_image_name': relevant_image_path,
                'bboxes': bbox_dict
                }
    result = json.dumps(result)
    
    
#     print(result["query_time"])
#     print(result["top_k_score"])
#     print(result["relevant_image_name"])
#     print(result["bboxs"])
    print(result)
    return result

def query_hash(query_img, net, topk, LSHash,getbbox=False):
    #query_img = cv2.imread(query_path)
    #cv2_imshow(resizeImg(query_img,0.20))
    start = time.time()
    query_feature = extractQueryFeature(query_img,net)
    query_feature = tonumpy(query_feature)
    
    res = LSHash.predict(query_feature)
    
    inx_ft = []
    for i in res:
        inx_ft.append(bdescs[i])
    inx_ft = tonumpy(np.array(inx_ft))
    query_feature = tonumpy(np.array(query_feature))
    scores = matmul(query_feature, inx_ft)
    ranklist = sorted(range(len(scores)), key=lambda i: scores[i])[-10:]
    ranklist = ranklist[::-1] #reverse
#     print(len(ranklist))
#     with open(os.environ['DIR_ROOT']+'test.txt', 'w') as f:
#         for item in ranklist:
#             f.write("%s\n" % res[item])
    query_time = time.time() - start
    
    
    #top_k_score = sorted(scores)[-1:-topk-1:-1]
    
#     relevant_image_path = []
#     relevant_image = []
#     for i in ranklist:
#         img_path = dataset.get_filename(res[i])
#         img = Image.open(img_path).convert('RGB')
#         image_name = img_path.split('/')[-1]
#         relevant_image_path.append(image_name)
    
#     result = {
#                 'query_time': str(query_time),
#                 #'top_k_score': str(top_k_score),
#                 'relevant_image_name': relevant_image_path
#                 }
#     print(result["query_time"])
#     #print(result["top_k_score"])
#     print(result["relevant_image_name"])
    top_k_score = [0.123] * topk
    
    relevant_image_path = []
    relevant_image = []
    
    for i in ranklist:
        img_path = dataset.get_filename(res[i])
        img = Image.open(img_path).convert('RGB')
        image_name = img_path.split('/')[-1]
        relevant_image_path.append(image_name)

        
    bbox = []
    num_match_keypoint = []
    #Spaital Matching
    if getbbox == True:
        q_kpts, q_descs, q_As, query_im_cv = AFFNetExtractor(img)
        for img_name in relevant_image_path:
            relevant_img = cv2.cvtColor(cv2.imread(os.path.join(os.environ['DB_ROOT'] + "/oxford5k/jpg",img_name)), cv2.COLOR_BGR2RGB)
            kpts, descs, As = load_npz(os.path.join(KEYPOINT_ROOT, img_name[:-4] + ".npz"))
            tentatives = match_snn(q_descs, descs, 0.85)
            if len(tentatives) >= 4:
                H, inl_mask = ransac_validate(tentatives, q_kpts, kpts, 2.0)  #
                box = draw_bbox(q_kpts, kpts, q_As, As, tentatives, inl_mask, query_im_cv, relevant_img, Hgt, H)
                num_match_keypoint.append(len(tentatives))
            else:
                box = np.zeros((4,2))
                num_match_keypoint.append(0)
                
            bbox.append(box.tolist())
            
        rerank = np.argsort(num_match_keypoint)[::-1]
        
        top_k_score = [top_k_score[i] for i in rerank]
        relevant_image_path = [relevant_image_path[i] for i in rerank]
        bbox = [bbox[i] for i in rerank]
    
    bbox_dict = {}
    print(bbox_dict)
    for r,b in enumerate(bbox):
        bbox_dict[r+1] = b
            
    query_time = time.time() - start
    
    result = {
                'query_time': round(query_time, 2),
                'top_k_score': str(top_k_score),
                'relevant_image_name': relevant_image_path,
                'bboxes': bbox_dict
                }
    result = json.dumps(result)
    
    return result

def query_another_hash(query_img, net, topk, LSHash,getbbox=False):
    #query_img = cv2.imread(query_path)
    #cv2_imshow(resizeImg(query_img,0.20))
    start = time.time()
    query_feature = extractQueryFeature(query_img,net)
    query_feature = tonumpy(query_feature)
    
    ranklist = LSHash.predict(query_feature)
    
#     inx_ft = []
#     for i in res:
#         inx_ft.append(bdescs[i])
#     inx_ft = tonumpy(np.array(inx_ft))
#     query_feature = tonumpy(np.array(query_feature))
#     scores = matmul(query_feature, inx_ft)
#     ranklist = sorted(range(len(scores)), key=lambda i: scores[i])[-10:]
#     ranklist = ranklist[::-1] #reverse
#     print(len(ranklist))
#     with open(os.environ['DIR_ROOT']+'test.txt', 'w') as f:
#         for item in ranklist:
#             f.write("%s\n" % res[item])
    query_time = time.time() - start
    
    
    #top_k_score = sorted(scores)[-1:-topk-1:-1]
    
#     relevant_image_path = []
#     relevant_image = []
#     for i in ranklist:
#         img_path = dataset.get_filename(res[i])
#         img = Image.open(img_path).convert('RGB')
#         image_name = img_path.split('/')[-1]
#         relevant_image_path.append(image_name)
    
#     result = {
#                 'query_time': str(query_time),
#                 #'top_k_score': str(top_k_score),
#                 'relevant_image_name': relevant_image_path
#                 }
#     print(result["query_time"])
#     #print(result["top_k_score"])
#     print(result["relevant_image_name"])
    top_k_score = [0.123] * topk
    
    relevant_image_path = []
    relevant_image = []
    
    for i in ranklist:
        img_path = dataset.get_filename(i)
        img = Image.open(img_path).convert('RGB')
        image_name = img_path.split('/')[-1]
        relevant_image_path.append(image_name)

        
    bbox = []
    num_match_keypoint = []
    #Spaital Matching
    if getbbox == True:
        q_kpts, q_descs, q_As, query_im_cv = AFFNetExtractor(img)
        for img_name in relevant_image_path:
            relevant_img = cv2.cvtColor(cv2.imread(os.path.join(os.environ['DB_ROOT'] + "/oxford5k/jpg",img_name)), cv2.COLOR_BGR2RGB)
            kpts, descs, As = load_npz(os.path.join(KEYPOINT_ROOT, img_name[:-4] + ".npz"))
            tentatives = match_snn(q_descs, descs, 0.85)
            if len(tentatives) >= 4:
                H, inl_mask = ransac_validate(tentatives, q_kpts, kpts, 2.0)  #
                box = draw_bbox(q_kpts, kpts, q_As, As, tentatives, inl_mask, query_im_cv, relevant_img, Hgt, H)
                num_match_keypoint.append(len(tentatives))
            else:
                box = np.zeros((4,2))
                num_match_keypoint.append(0)
                
            bbox.append(box.tolist())
            
        rerank = np.argsort(num_match_keypoint)[::-1]
        
        top_k_score = [top_k_score[i] for i in rerank]
        relevant_image_path = [relevant_image_path[i] for i in rerank]
        bbox = [bbox[i] for i in rerank]
    
    bbox_dict = {}
    print(bbox_dict)
    for r,b in enumerate(bbox):
        bbox_dict[r+1] = b
            
    query_time = time.time() - start
    
    result = {
                'query_time': round(query_time, 2),
                'top_k_score': str(top_k_score),
                'relevant_image_name': relevant_image_path,
                'bboxes': bbox_dict
                }
    result = json.dumps(result)
    
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
    return "<div>Server is running, now you can connect to our backend services. Visit our main page at <a href='https://cs-336-image-retrieval.vercel.app/image-retrieval'>this link<a/></div>"

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
    ['christ_church', 'trinity', 'magdalen', 'Roxford', 'balliol', 'hertford', 'radcliffe_camera', 'new', 'ashmolean', 'all_souls', 'oriel', 'worcester', 'bodleian', 'cornmarket', 'pitt_rivers', 'keble', 'jesus']
    """
    
    category = request.form.get('category')
    
    
    if category in image_dict.keys():
        print (f'Return response for {category}')
        result = image_dict[category][:120]
        random.shuffle(result)
        result = result[:20]
    else:
        print (f'Return random images, query category: {category}')
        query_set = ['all_souls_000013.jpg', 'all_souls_000026.jpg', 'oxford_002985.jpg', 'all_souls_000051.jpg', 'oxford_003410.jpg', 'ashmolean_000058.jpg', 'ashmolean_000000.jpg', 'ashmolean_000269.jpg', 'ashmolean_000007.jpg', 'ashmolean_000305.jpg', 'balliol_000051.jpg', 'balliol_000187.jpg', 'balliol_000167.jpg', 'balliol_000194.jpg', 'oxford_001753.jpg', 'bodleian_000107.jpg', 'bodleian_000407.jpg', 'bodleian_000163.jpg', 'christ_church_000179.jpg', 'oxford_002734.jpg', 'christ_church_000999.jpg', 'christ_church_001020.jpg', 'oxford_002562.jpg', 'cornmarket_000047.jpg', 'cornmarket_000105.jpg', 'cornmarket_000019.jpg', 'oxford_000545.jpg', 'cornmarket_000131.jpg', 'hertford_000015.jpg', 'oxford_001752.jpg', 'oxford_000317.jpg', 'hertford_000027.jpg', 'hertford_000063.jpg', 'keble_000245.jpg', 'keble_000214.jpg', 'keble_000227.jpg', 'keble_000028.jpg', 'keble_000055.jpg', 'magdalen_000078.jpg','oxford_003335.jpg', 'magdalen_000058.jpg', 'oxford_001115.jpg', 'magdalen_000560.jpg', 'pitt_rivers_000033.jpg', 'pitt_rivers_000119.jpg', 'pitt_rivers_000153.jpg', 'pitt_rivers_000087.jpg','pitt_rivers_000058.jpg', 'radcliffe_camera_000519.jpg', 'oxford_002904.jpg', 'radcliffe_camera_000523.jpg', 'radcliffe_camera_000095.jpg', 'bodleian_000132.jpg']
        result = random.sample(query_set,20)
        #result = random.sample(os.listdir(os.path.join(os.environ['DB_ROOT'] + "/oxford5k/jpg")), 20)
        
        
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
    if (request.form.get('lsh') == 'true'):
        result = query_another_hash(query_img, net, int(request.form.get('top_k')), LSH)    
    else:
        result = query(query_img ,net , int(request.form.get('top_k')), request.form.get('return_bboxes') == 'true')
    
    
    #Return result to Client
    response = {
        'status': 200,
        'message': 'OK',
        'result': result,
    }
    
    # add header

    return(response)

# @app.route('/query', methods=['POST'])
# @cross_origin(origin='*',headers=['Content-Type','Authorization'])
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
#             top_k_score: (list of float) similarity score of relevant images
#             relevant_image_name: (list of string) name of relevant images
#         }
#     """
    
#     #Read Image form client and convert to PIL Image
#     query_img = request.form.get('query')
#     top_k = request.form.get('top_k')
#     getbbox = request.form.get('return_bboxes') == 'true'
    
#     query_img = base64toPIL(query_img)
#     query_img = transforms_img(query_img).unsqueeze(0)
    
#     #Query
#     result = query(query_img,net,10,getbbox)
    
#     #Return result to Client
#     response = {
#         'status': 200,
#         'message': 'OK',
#         'result': result,
#     }
    
#     # add header

#     return(response)


if __name__ == "__main__":
    
    #Load Network
    net = load_model(os.path.join(os.environ['DIR_ROOT'],"checkpoints/Resnet-101-AP-GeM.pt"),False)

    #Load data
    dataset = ImageListRelevants(os.environ["DB_ROOT"]+"/oxford5k/gnd_oxford5k.pkl", os.environ["DB_ROOT"] + "/oxford5k", "jpg" )
    dataLoader = get_loader(dataset, trf_chain="", preprocess=net.preprocess, iscuda=True,
                            output=['img'], batch_size=1, threads=1, shuffle=False)
    #Load database feature
    bdescs = np.load(os.path.join(os.environ['DB_ROOT'] + "/oxford5k", 'feats.bdescs.npy'))
    #LSHash = LSH(bdescs)
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
    LSH = anotherLSH(bdescs)
    
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

    #Load AFFNet Extractor
    hardnet = KF.HardNet(True).to('cpu').eval()
    affnet = KF.PatchAffineShapeEstimator(32).to('cpu')
    orinet = torch.jit.load('/home/jovyan/work/ImageRetrival/affnet/OriNetJIT.pt')
    orinet.eval()
    Hgt = np.loadtxt('/home/jovyan/work/ImageRetrival/affnet/H1to6p')
    
    app.run(host='0.0.0.0', port='6868')