# Comsen2026寒假任务算法方向——零样本目标检测
任务一：GroundingDINO复现

1.在开源仓库中下载GroundingDINO源码；

2.运行inference_on_a_image.py，用一张图片初步验证模型是否加载成功；

3.用coco/val2017和coco/annotations数据集进行整体模型的评估。

推理与评测一键运行说明：
python run_detection_final.py      # 推理检测
python visualize_analysis.py       # 可视化置信度图表
python visualize_mAP.py            # 可视化mAP相关图表
python evaluate.py                 # 评测

任务二：零样本设置与数据集

1.选择MS COCO数据集并采用经典划分“65seen/15unseen”；

2.用划分的数据集训练和评估GroundingDINO基线模型。
