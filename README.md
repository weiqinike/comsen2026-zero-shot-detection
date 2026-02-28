# Comsen2026寒假任务算法方向——零样本目标检测
任务一：GroundingDINO复现

1.在开源仓库中下载GroundingDINO源码；

2.运行inference_on_a_image.py，用一张图片初步验证模型是否加载成功；

3.用coco/val2017和coco/annotations数据集进行整体模型的评估。

环境运行补充说明：

在运行GroundingDINO模型时，构造了一个虚拟环境，并将需要配置的东西下载在该虚拟环境中。

启动该虚拟环境：

conda activate groundingdino

推理与评测一键运行说明：

python run_detection_final.py      # 推理检测

python visualize_analysis.py       # 可视化置信度图表

python visualize_mAP.py            # 可视化mAP相关图表

python evaluate.py                 # 评测

任务二：零样本设置与数据集

1.选择MS COCO数据集并采用经典划分“65seen/15unseen”；

2.用划分的数据集训练和评估GroundingDINO基线模型。

seen和unseen类别说明

seen_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop','teddy bear'
        ]

unseen_classes = [
            'toaster', 'hair drier', 'mouse', 'microwave', 'scissors',
            'oven', 'clock', 'book', 'refrigerator', 'toothbrush',
            'vase', 'remote', 'keyboard', 'cell phone', 'sink'
        ]

PS.第一次检测和第四次检测未生成JSON文件检测结果

任务三：提示词工程与对比实验

1.选择person、car、chair三个类别；

2.使用手动计算AP的方式；

3.选择方向A进行改进。

PS.实验报告传到github中图片显示不出来了，我放了一个文件夹，然后实验报告的markdown文档我私发给林小扬学长。
