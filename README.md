<details open>
<summary>Install</summary>

Python, PyTorch등 버전은 requirements.txt를 참조하세요
<!-- $ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev -->

```bash
$ git clone https://github.com/ultralytics/yolov5
$ cd yolov5
$ pip install -r requirements.txt
```

</details>

<details open>


<details>
<summary>사용법</summary>

`train.py`에서 학습을 진행하고 `detect.py` 에서 객체를 탐지합니다. 다양한 파라미터들은 의도에 맞게 적절히 수정해야 합니다.

```bash
$ python train.py --img 640 --batch 16 --epochs 100 --data datasets/sign/data.yaml --weights yolov5s.pt

$ python detect.py --weights ./runs/train/exp/weights/best.pt --imgsz 640							
```

</details> 

## <div align="center">환경</div>

<div align="center">
    <a href="https://www.python.org/">
        <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" width="15%"/>
    </a>
    <a href="https://pytorch.org/">
        <img src="https://upload.wikimedia.org/wikipedia/commons/9/96/Pytorch_logo.png" width="15%"/>
    </a>
    <a href="https://www.tensorflow.org/">
        <img src="https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg" width="15%"/>
    </a>
    <a href="https://opencv.org/">
        <img src="https://upload.wikimedia.org/wikipedia/commons/3/32/OpenCV_Logo_with_text.png" width="15%"/>
    </a>
</div>

## <div align="center">YOLOv5의 기본적인 참고자료</div>

<p align="center"><img width="800" src="https://user-images.githubusercontent.com/26833433/114313216-f0a5e100-9af5-11eb-8445-c682b60da2e3.png"></p>
<details>
  <summary>YOLOv5-P5 640 Figure (click to expand)</summary>

<p align="center"><img width="800" src="https://user-images.githubusercontent.com/26833433/114313219-f1d70e00-9af5-11eb-9973-52b1f98d321a.png"></p>
</details>
<details>
  <summary>Figure Notes (click to expand)</summary>

* GPU Speed measures end-to-end time per image averaged over 5000 COCO val2017 images using a V100 GPU with batch size
  32, and includes image preprocessing, PyTorch FP16 inference, postprocessing and NMS.
* EfficientDet data from [google/automl](https://github.com/google/automl) at batch size 8.
* **Reproduce** by
  `python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5s6.pt yolov5m6.pt yolov5l6.pt yolov5x6.pt`

</details>

[assets]: https://github.com/ultralytics/yolov5/releases

|Model |size<br><sup>(pixels) |mAP<sup>val<br>0.5:0.95 |mAP<sup>test<br>0.5:0.95 |mAP<sup>val<br>0.5 |Speed<br><sup>V100 (ms) | |params<br><sup>(M) |FLOPs<br><sup>640 (B)
|---                    |---  |---      |---      |---      |---     |---|---   |---
|[YOLOv5s][assets]      |640  |36.7     |36.7     |55.4     |**2.0** |   |7.3   |17.0
|[YOLOv5m][assets]      |640  |44.5     |44.5     |63.1     |2.7     |   |21.4  |51.3
|[YOLOv5l][assets]      |640  |48.2     |48.2     |66.9     |3.8     |   |47.0  |115.4
|[YOLOv5x][assets]      |640  |**50.4** |**50.4** |**68.8** |6.1     |   |87.7  |218.8
|                       |     |         |         |         |        |   |      |
|[YOLOv5s6][assets]     |1280 |43.3     |43.3     |61.9     |**4.3** |   |12.7  |17.4
|[YOLOv5m6][assets]     |1280 |50.5     |50.5     |68.7     |8.4     |   |35.9  |52.4
|[YOLOv5l6][assets]     |1280 |53.4     |53.4     |71.1     |12.3    |   |77.2  |117.7
|[YOLOv5x6][assets]     |1280 |**54.4** |**54.4** |**72.0** |22.4    |   |141.8 |222.9
|                       |     |         |         |         |        |   |      |
|[YOLOv5x6][assets] TTA |1280 |**55.0** |**55.0** |**72.0** |70.8    |   |-     |-

<details>
  <summary>Table Notes (click to expand)</summary>

* AP<sup>test</sup> denotes COCO [test-dev2017](http://cocodataset.org/#upload) server results, all other AP results
  denote val2017 accuracy.
* AP values are for single-model single-scale unless otherwise noted. **Reproduce mAP**
  by `python val.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65`
* Speed<sub>GPU</sub> averaged over 5000 COCO val2017 images using a
  GCP [n1-standard-16](https://cloud.google.com/compute/docs/machine-types#n1_standard_machine_types) V100 instance, and
  includes FP16 inference, postprocessing and NMS. **Reproduce speed**
  by `python val.py --data coco.yaml --img 640 --conf 0.25 --iou 0.45 --half`
* All checkpoints are trained to 300 epochs with default settings and hyperparameters (no autoaugmentation).
* Test Time Augmentation ([TTA](https://github.com/ultralytics/yolov5/issues/303)) includes reflection and scale
  augmentation. **Reproduce TTA** by `python val.py --data coco.yaml --img 1536 --iou 0.7 --augment`

</details>

## <div align="center">레퍼런스</div>
* [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)&nbsp;
* [Tips for Best Training Results](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)&nbsp;
* [Multi-GPU Training](https://github.com/ultralytics/yolov5/issues/475)
* [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36)&nbsp;
* [Model Pruning/Sparsity](https://github.com/ultralytics/yolov5/issues/304)
* [Transfer Learning with Frozen Layers](https://github.com/ultralytics/yolov5/issues/1314)&nbsp;
* [TensorRT Deployment](https://github.com/wang-xinyu/tensorrtx)
* [yolov4 학습](https://velog.io/@jhlee508/Object-Detection-YOLOv4-Darknet-%ED%95%99%EC%8A%B5%ED%95%98%EC%97%AC-Custom-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%9D%B8%EC%8B%9D-%EB%AA%A8%EB%8D%B8-%EB%A7%8C%EB%93%A4%EA%B8%B0-feat.-AlexeyABdarknet)
* [오픈소스 수어번역기](https://github.com/23bulgogi/sonmari)

