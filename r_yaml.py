#출처: https://rfriend.tistory.com/540 [R, Python 분석과 프로그래밍의 친구 (by R Friend)]
import yaml
"""
Traffic_Light = {'names': ['Traffic_Light'], 'nc': 1, 'path': '../datasets/data', 'test': 'train.txt', 'train': 'train.txt', 'val': 'train.txt'}
with open('data/Traffic_Light.yaml', 'w') as f:
    yaml.dump(Traffic_Light, f)

with open('data/Traffic_Light.yaml', encoding='utf8') as f:
    coco = yaml.load(f, Loader=yaml.FullLoader)
    #print(coco)
    print("---------------------")
    # sorting by Key
    coco_sorted = yaml.dump(coco, sort_keys=True)
    print(coco_sorted)
"""
#with open('data/Traffic_Light.yaml', encoding='utf8') as f:
with open('../datasets/mushroom/data.yaml', encoding='utf8') as f:
    coco = yaml.load(f, Loader=yaml.FullLoader)
    #print(coco)
    print("---------------------")
    # sorting by Key
    coco_sorted = yaml.dump(coco, sort_keys=True)
    print(coco_sorted)
