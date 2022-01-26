# ImageRetrival

#Query API
""/query", methods=['POST']
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
    

 
#Get suggest image API
"/get-suggest-query",methods=['GET']
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


#Get Image API
"/get-image/<image_name>",methods=['GET']