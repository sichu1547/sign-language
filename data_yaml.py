# -*- coding: utf-8 -*-
import yaml
from glob import glob

n = 'sign'

with open('datasets/'+n+'/data.yaml', 'r', encoding='UTF8') as f:
	data = yaml.safe_load(f)    
print(data)

data['train'] = 'datasets/'+n+'/train/images'
data['test'] = 'datasets/'+n+'/test/images'
data['val'] = 'datasets/'+n+'/valid/images'

data['nc'] = 6

data['names'] = ['sick', 'cold', 'four', 'nose', 'fever1', 'fever2']

with open('datasets/'+n+'/data.yaml', 'w') as f:
  yaml.dump(data, f)
print(data)
    

