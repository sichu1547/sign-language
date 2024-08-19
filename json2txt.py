# -*- coding: utf-8 -*-
import json
import os

d = os.listdir('ori/dataset/labels_j/')
for i in d:
    with open('ori/dataset/labels_j/'+i, 'r') as f:
        json_data = json.load(f)
        disease = json_data['annotations']['disease']
        h = json_data['description']['height']
        w = json_data['description']['width']        
        #len(json_data['annotations']['points'])
        xtl = json_data['annotations']['points'][0]['xtl']/w
        ytl = json_data['annotations']['points'][0]['ytl']/h
        xbr = json_data['annotations']['points'][0]['xbr']/w
        ybr = json_data['annotations']['points'][0]['ybr']/h
        label = [disease, xtl, ytl, xbr, ybr]        
        f_n = 'ori/dataset/labels/'+i.split('.json')[0].split('.jpg')[0].split('.JPG')[0].split('.jpeg')[0].split('.JPEG')[0]+'.txt'        
        with open(f_n, "w") as text_file:
            for l in label:
                text_file.write(str(l)+' ')