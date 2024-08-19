import os
import shutil

path = '../datasets/data/img'
lpath = '../datasets/data/labels'
ipath = '../datasets/data/images'
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pli = os.listdir(path)

for i in pli:
    fname, ext = os.path.splitext(path + '/' + i)
    if ext=='.txt' :
        print(i)
        shutil.copy(path + '/' + i, lpath + '/' + i)

    elif ext=='.jpg' :
        print(i)
        shutil.copy(path + '/' + i, ipath + '/' + i)
