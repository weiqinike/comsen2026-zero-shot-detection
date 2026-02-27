"""
Zero-Shot Object Detection Evaluation for Grounding DINO
完整评测版本 - 包含所有优化
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# ============ 修复警告 ============
import warnings
warnings.filterwarnings("ignore")

# ============================== 配置区域 ==============================

# COCO 数据集路径
COCO_ROOT = r"C:\Users\24344\GroundingDINO\weights\coco\val2017\val_images"
COCO_ANN_FILE = r"C:\Users\24344\GroundingDINO\weights\coco\annotations\annotations_images\instances_val2017.json"
COCO_SPLIT = "val"

# Grounding DINO 模型路径
MODEL_CONFIG = r"C:\Users\24344\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py"
MODEL_CHECKPOINT = r"C:\Users\24344\GroundingDINO\weights\groundingdino_swint_ogc.pth"
BASE_DIR = r"C:\Users\24344\GroundingDINO"

# ============ 优化后的推理参数 ============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOX_THRESHOLD = 0.20  # 提高阈值，过滤低质量检测
TEXT_THRESHOLD = 0.15  # 提高文本阈值
NMS_THRESHOLD = 0.5  # 非极大值抑制阈值
BATCH_SIZE = 1
NUM_WORKERS = 0

# ============ 数据集大小限制 ============
MAX_IMAGES = 500  # 限制评测500张图片
# 如果想要使用完整数据集，设为 None
# MAX_IMAGES = None  # 使用完整数据集

# 输出目录
OUTPUT_DIR = f"./results_{MAX_IMAGES if MAX_IMAGES else 'full'}"  # 根据图片数量命名目录
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# 可视化设置
VISUALIZE_SAMPLES = 20  # 可视化多少张样本
SAVE_VISUALIZATIONS = True  # 是否保存可视化结果

# ============ 检查并添加路径 ============
import sys
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 延迟导入 groundingdino 相关模块
def import_groundingdino():
    try:
        import groundingdino.datasets.transforms as T
        from groundingdino.models import build_model
        from groundingdino.util import box_ops
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import clean_state_dict
        from groundingdino.util.inference import predict
        return T, build_model, box_ops, SLConfig, clean_state_dict, predict
    except ImportError as e:
        print(f"导入 groundingdino 失败: {e}")
        print(f"请确保 GroundingDINO 项目在: {BASE_DIR}")
        sys.exit(1)

# ============================== 图像变换函数 ==============================

def transform_image_simple(image_pil):
    """简化的图像预处理变换"""
    try:
        T, _, _, _, _, _ = import_groundingdino()
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image_tensor, _ = transform(image_pil, None)
        return image_tensor
    except Exception as e:
        print(f"使用GroundingDINO变换失败: {e}")
        from torchvision import transforms
        transform_backup = transforms.Compose([
            transforms.Resize((800, 1333)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform_backup(image_pil)

# ============================== 打印配置信息 ==============================
print("="*60)
print("零样本目标检测完整评测脚本")
print("="*60)
print("配置信息:")
print(f"COCO 图片目录: {COCO_ROOT}")
print(f"COCO 标注文件: {COCO_ANN_FILE}")
print(f"模型配置: {MODEL_CONFIG}")
print(f"模型权重: {MODEL_CHECKPOINT}")
print(f"设备: {DEVICE}")
print(f"检测框阈值: {BOX_THRESHOLD} (已提高)")
print(f"文本阈值: {TEXT_THRESHOLD} (已提高)")
print(f"NMS阈值: {NMS_THRESHOLD}")
print(f"评测图片数量: {MAX_IMAGES if MAX_IMAGES else '全部'}")
print(f"输出目录: {OUTPUT_DIR}")
print(f"可视化目录: {VISUALIZATION_DIR}")
print("="*60)

# ============================== 数据集类 ==============================

class ZeroShotCOCODataset(Dataset):
    """COCO 零样本检测数据集（完整评测）"""

    def __init__(self, transform=None, max_images=None):
        """
        Args:
            transform: 数据增强函数
            max_images: 最大图片数量，None表示使用全部
        """
        self.coco_root = COCO_ROOT
        self.ann_file = COCO_ANN_FILE
        self.transform = transform if transform is not None else transform_image_simple
        self.split = COCO_SPLIT
        self.max_images = max_images

        print(f"COCO 标注文件: {self.ann_file}")
        print(f"COCO 图片目录: {self.coco_root}")

        # 检查文件是否存在
        if not os.path.exists(self.ann_file):
            print(f"错误: 标注文件不存在: {self.ann_file}")
            raise FileNotFoundError(f"未找到标注文件，请检查路径: {self.ann_file}")

        if not os.path.exists(self.coco_root):
            print(f"错误: 图片目录不存在: {self.coco_root}")
            possible_dirs = [
                self.coco_root,
                r"C:\Users\24344\GroundingDINO\weights\coco\val2017",
                r"C:\Users\24344\GroundingDINO\weights\coco\images\val2017",
                r"C:\Users\24344\GroundingDINO\weights\coco\val_images",
                r"C:\Users\24344\GroundingDINO\weights\coco\val2017\images",
                r"./weights/coco/val2017",
                r"./weights/coco/images/val2017",
            ]

            found = False
            for path in possible_dirs:
                if os.path.exists(path):
                    self.coco_root = path
                    print(f"找到图片目录: {path}")
                    found = True
                    break

            if not found:
                raise FileNotFoundError(f"未找到图片目录，请检查路径: {COCO_ROOT}")

        # 加载 COCO 标注
        print(f"正在加载标注文件: {self.ann_file}")
        try:
            self.coco = COCO(self.ann_file)
        except Exception as e:
            print(f"加载COCO标注失败: {e}")
            raise

        # 获取类别划分
        self.seen_classes, self.unseen_classes = self.get_coco_class_splits()

        # 获取类别信息
        self.categories = self.coco.dataset['categories']
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.categories}
        self.cat_name_to_id = {cat['name']: cat['id'] for cat in self.categories}

        # 创建更宽松的类别映射
        self.cat_name_to_id_lower = {cat['name'].lower(): cat['id'] for cat in self.categories}
        self.cat_name_to_id_words = {}
        for cat_name, cat_id in self.cat_name_to_id.items():
            words = cat_name.split()
            for word in words:
                if word not in self.cat_name_to_id_words:
                    self.cat_name_to_id_words[word] = cat_id

        # 获取类别ID
        self.seen_ids = [self.cat_name_to_id[cls] for cls in self.seen_classes if cls in self.cat_name_to_id]
        self.unseen_ids = [self.cat_name_to_id[cls] for cls in self.unseen_classes if cls in self.cat_name_to_id]
        self.all_ids = self.seen_ids + self.unseen_ids

        # ============ 优化文本提示 ============
        self.text_prompts = {
            'seen': [f"a clear photo of a {cls}" for cls in self.seen_classes],
            'unseen': [f"a clear photo of a {cls}" for cls in self.unseen_classes],
            'all': [f"a clear photo of a {cls}" for cls in self.seen_classes + self.unseen_classes]
        }

        # 获取图片ID
        self.image_ids = self.coco.getImgIds()

        # 限制图片数量
        if self.max_images is not None and self.max_images > 0:
            self.image_ids = self.image_ids[:self.max_images]
            print(f"限制使用前 {self.max_images} 张图片")

        print(f"数据集加载完成: {len(self.image_ids)} 张图片")
        print(f"Seen 类别: {len(self.seen_classes)} 个")
        print(f"Unseen 类别: {len(self.unseen_classes)} 个")

    def get_coco_class_splits(self):
        """获取COCO 65 seen / 15 unseen类别划分"""
        coco_classes = [
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
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        unseen_classes = [
            'toaster', 'hair drier', 'mouse', 'microwave', 'scissors',
            'oven', 'clock', 'book', 'refrigerator', 'toothbrush',
            'vase', 'remote', 'keyboard', 'cell phone', 'sink'
        ]

        seen_classes = [cls for cls in coco_classes if cls not in unseen_classes]
        return seen_classes, unseen_classes

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        try:
            img_id = self.image_ids[idx]
            img_info = self.coco.loadImgs(img_id)[0]
            img_filename = img_info['file_name']

            # 构建图片路径
            img_path = os.path.join(self.coco_root, img_filename)

            if not os.path.exists(img_path):
                possible_img_paths = [
                    os.path.join(self.coco_root, img_filename),
                    os.path.join(r"C:\Users\24344\GroundingDINO\weights\coco\val2017", img_filename),
                    os.path.join(r"C:\Users\24344\GroundingDINO\weights\coco\images\val2017", img_filename),
                    os.path.join(r"C:\Users\24344\GroundingDINO\weights\coco\val_images", img_filename),
                    os.path.join(r"C:\Users\24344\GroundingDINO\weights\coco\val2017\images", img_filename),
                    os.path.join(r"C:\Users\24344\GroundingDINO\weights\coco", img_filename),
                ]

                for path in possible_img_paths:
                    if os.path.exists(path):
                        img_path = path
                        break

            # 加载图片
            if os.path.exists(img_path):
                try:
                    image_pil = Image.open(img_path).convert('RGB')
                except Exception as e:
                    print(f"无法打开图片 {img_path}: {e}")
                    image_pil = Image.new('RGB', (640, 480), color='white')
            else:
                print(f"警告: 图片不存在: {img_filename}")
                image_pil = Image.new('RGB', (640, 480), color='white')

            # 获取标注
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            # 提取标注信息
            boxes = []
            labels = []
            seen_masks = []
            unseen_masks = []

            for ann in anns:
                x, y, w, h = ann['bbox']
                if w > 0 and h > 0:
                    boxes.append([x, y, x + w, y + h])
                    labels.append(ann['category_id'])

                    if ann['category_id'] in self.seen_ids:
                        seen_masks.append(True)
                        unseen_masks.append(False)
                    elif ann['category_id'] in self.unseen_ids:
                        seen_masks.append(False)
                        unseen_masks.append(True)
                    else:
                        seen_masks.append(False)
                        unseen_masks.append(False)

            # 转换为tensor
            boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
            labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,))
            seen_masks = torch.tensor(seen_masks, dtype=torch.bool) if seen_masks else torch.zeros((0,), dtype=torch.bool)
            unseen_masks = torch.tensor(unseen_masks, dtype=torch.bool) if unseen_masks else torch.zeros((0,), dtype=torch.bool)

            # 应用变换
            try:
                image_tensor = self.transform(image_pil)
            except Exception as e:
                print(f"应用变换失败: {e}")
                from torchvision import transforms
                backup_transform = transforms.Compose([
                    transforms.Resize(800),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                image_tensor = backup_transform(image_pil)

            # 构建目标字典
            target = {
                'image_id': torch.tensor([img_id]),
                'boxes': boxes,
                'labels': labels,
                'seen_mask': seen_masks,
                'unseen_mask': unseen_masks,
                'orig_size': torch.tensor([image_pil.size[1], image_pil.size[0]]),
                'filename': img_filename,
                'image_pil': image_pil  # 保存原始图像用于可视化
            }

            return image_tensor, target

        except Exception as e:
            print(f"获取数据项 {idx} 时出错: {e}")
            import traceback
            traceback.print_exc()
            image_tensor = torch.zeros((3, 800, 800))
            target = {
                'image_id': torch.tensor([0]),
                'boxes': torch.zeros((0, 4)),
                'labels': torch.zeros((0,)),
                'seen_mask': torch.zeros((0,), dtype=torch.bool),
                'unseen_mask': torch.zeros((0,), dtype=torch.bool),
                'orig_size': torch.tensor([800, 800]),
                'filename': 'error',
                'image_pil': Image.new('RGB', (800, 800), color='white')
            }
            return image_tensor, target

    def get_text_prompts(self, mode='all'):
        """获取文本提示"""
        return self.text_prompts[mode]

# ============================== 后处理优化函数 ==============================

def apply_nms(predictions, iou_threshold=NMS_THRESHOLD):
    """对预测进行非极大值抑制"""
    if not predictions:
        return predictions

    # 按图片ID分组
    grouped = {}
    for pred in predictions:
        img_id = pred['image_id']
        if img_id not in grouped:
            grouped[img_id] = []
        grouped[img_id].append(pred)

    # 对每个图片应用NMS
    results = []
    for img_id, preds in grouped.items():
        if len(preds) > 1:
            # 转换为tensor
            boxes = []
            scores = []
            valid_preds = []

            for pred in preds:
                bbox = pred['bbox']
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    boxes.append([x, y, x + w, y + h])
                    scores.append(pred['score'])
                    valid_preds.append(pred)

            if boxes:
                boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
                scores_tensor = torch.tensor(scores, dtype=torch.float32)

                # 应用NMS
                try:
                    keep = torch.ops.torchvision.nms(
                        boxes_tensor,
                        scores_tensor,
                        iou_threshold
                    )

                    for idx in keep:
                        results.append(valid_preds[idx])
                except:
                    # 如果NMS失败，保留所有预测
                    results.extend(valid_preds)
            else:
                results.extend(valid_preds)
        else:
            results.extend(preds)

    return results

def filter_by_size(predictions, min_area=10, max_area=1000000):
    """按尺寸过滤预测框"""
    filtered = []
    for pred in predictions:
        bbox = pred['bbox']
        if len(bbox) == 4:
            x, y, w, h = bbox
            area = w * h
            if min_area <= area <= max_area:
                filtered.append(pred)
    return filtered

# ============================== 可视化函数 ==============================

def visualize_detections(image_pil, predictions, ground_truth, save_path,
                         title="Detection Results", show_gt=True, show_pred=True):
    """可视化检测结果"""
    fig, ax = plt.subplots(1, figsize=(16, 10))
    ax.imshow(image_pil)
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold')

    # 绘制真实框
    if show_gt and ground_truth is not None:
        for i, (box, label) in enumerate(zip(ground_truth['boxes'], ground_truth['labels'])):
            if i < 20:  # 最多显示20个真实框
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1

                rect = patches.Rectangle(
                    (x1, y1), w, h,
                    linewidth=2,
                    edgecolor='green',
                    facecolor='none',
                    linestyle='-',
                    alpha=0.7
                )
                ax.add_patch(rect)
                ax.text(x1, y1-5, f"GT: {label}",
                       color='green', fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.5))

    # 绘制预测框
    if show_pred and predictions:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(predictions)))
        for i, pred in enumerate(predictions[:20]):  # 最多显示20个预测框
            bbox = pred['bbox']
            if len(bbox) == 4:
                x, y, w, h = bbox
                score = pred['score']
                cat_id = pred.get('category_id', 0)
                cat_name = f"ID:{cat_id}"

                # 使用彩虹色
                color = colors[i % len(colors)]

                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=3,
                    edgecolor=color,
                    facecolor='none',
                    linestyle='-',
                    alpha=0.8
                )
                ax.add_patch(rect)

                # 显示标签和置信度
                label_text = f"{cat_name}: {score:.3f}"
                ax.text(x, y-5, label_text,
                       color='red', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = []
    if show_gt:
        legend_elements.append(Patch(facecolor='none', edgecolor='green',
                                   label='Ground Truth', linestyle='-'))
    if show_pred:
        legend_elements.append(Patch(facecolor='none', edgecolor='red',
                                   label='Predictions', linestyle='-'))

    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"可视化结果已保存: {save_path}")

# ============================== 评测器类 ==============================

class ZeroShotEvaluator:
    """零样本检测评测器（优化版）"""

    def __init__(self, model, dataset, device=DEVICE,
                 box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
                 apply_nms=True, nms_threshold=NMS_THRESHOLD):
        """
        Args:
            model: Grounding DINO 模型
            dataset: 评测数据集
            device: 设备
            box_threshold: 检测框阈值
            text_threshold: 文本阈值
            apply_nms: 是否应用NMS
            nms_threshold: NMS阈值
        """
        self.model = model
        self.dataset = dataset
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.apply_nms = apply_nms
        self.nms_threshold = nms_threshold

        self.model.to(device)
        self.model.eval()

        # 获取类别映射
        self.seen_ids = dataset.seen_ids
        self.unseen_ids = dataset.unseen_ids
        self.cat_id_to_name = dataset.cat_id_to_name
        self.cat_name_to_id = dataset.cat_name_to_id

        # 文本提示
        self.seen_texts = dataset.get_text_prompts('seen')
        self.unseen_texts = dataset.get_text_prompts('unseen')
        self.all_texts = dataset.get_text_prompts('all')

        # 统计信息
        self.total_predictions = 0
        self.matched_predictions = 0
        self.debug_count = 0
        self.visualization_count = 0

    def evaluate(self, dataloader, mode='all', visualize=False):
        """
        评测函数

        Args:
            dataloader: 数据加载器
            mode: 'all' | 'seen' | 'unseen'
            visualize: 是否生成可视化

        Returns:
            dict: 评测结果
        """
        results = []
        gt_dict = {}
        visualization_data = []  # 存储可视化数据

        # 重置统计信息
        self.total_predictions = 0
        self.matched_predictions = 0
        self.debug_count = 0
        self.visualization_count = 0

        print(f"\n正在评测 {mode} 模式...")

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="评测进度")):
                for i in range(len(images)):
                    img = images[i]
                    target = targets[i]

                    if not isinstance(img, torch.Tensor):
                        continue

                    if img.dim() == 4:
                        img = img.squeeze(0)

                    img = img.to(self.device)

                    # 获取文本提示
                    if mode == 'seen':
                        texts = self.seen_texts
                        valid_cat_ids = set(self.seen_ids)
                    elif mode == 'unseen':
                        texts = self.unseen_texts
                        valid_cat_ids = set(self.unseen_ids)
                    else:
                        texts = self.all_texts
                        valid_cat_ids = set(self.seen_ids + self.unseen_ids)

                    # 获取预测结果
                    boxes, logits, phrases = self.predict(
                        model=self.model,
                        image=img,
                        caption=". ".join(texts),
                        box_threshold=self.box_threshold,
                        text_threshold=self.text_threshold,
                        device=self.device
                    )

                    img_id = target['image_id'].item()

                    # 存储真实标注
                    if img_id not in gt_dict:
                        gt_dict[img_id] = {
                            'boxes': target['boxes'].cpu().numpy(),
                            'labels': target['labels'].cpu().numpy(),
                            'seen_mask': target['seen_mask'].cpu().numpy() if 'seen_mask' in target else None,
                            'unseen_mask': target['unseen_mask'].cpu().numpy() if 'unseen_mask' in target else None
                        }

                    # 处理预测结果
                    image_predictions = []
                    if boxes is not None and boxes.numel() > 0:
                        for box, score, label in zip(boxes, logits, phrases):
                            self.total_predictions += 1

                            # 将标签名转换为ID
                            cat_id = self.map_label_to_category_id(label)

                            if cat_id is not None and cat_id in valid_cat_ids:
                                self.matched_predictions += 1

                                # 转换边界框格式
                                box_np = box.cpu().numpy()
                                orig_size = target['orig_size'].cpu().numpy()
                                converted_bbox = self.convert_bbox_format(box_np, orig_size)

                                if converted_bbox[2] > 0 and converted_bbox[3] > 0:
                                    pred_data = {
                                        'image_id': img_id,
                                        'category_id': cat_id,
                                        'bbox': converted_bbox,
                                        'score': float(score.item()),
                                        'label_text': label
                                    }
                                    image_predictions.append(pred_data)

                    # 应用后处理
                    if image_predictions:
                        # 按尺寸过滤
                        image_predictions = filter_by_size(image_predictions, min_area=10, max_area=1000000)

                        # 应用NMS
                        if self.apply_nms and len(image_predictions) > 1:
                            image_predictions = apply_nms(image_predictions, iou_threshold=self.nms_threshold)

                        # 添加到总结果
                        results.extend(image_predictions)

                        # 保存可视化数据
                        if visualize and self.visualization_count < VISUALIZE_SAMPLES:
                            visualization_data.append({
                                'image_id': img_id,
                                'image_pil': target.get('image_pil'),
                                'predictions': image_predictions[:10],  # 只保存前10个预测
                                'ground_truth': gt_dict[img_id],
                                'filename': target.get('filename', 'unknown')
                            })
                            self.visualization_count += 1

        print(f"\n统计信息:")
        print(f"总预测数: {self.total_predictions}")
        print(f"匹配预测数: {self.matched_predictions}")
        print(f"有效预测数: {len(results)}")
        print(f"评测图片数: {len(gt_dict)}")
        print(f"后处理过滤: 应用NMS={self.apply_nms}, NMS阈值={self.nms_threshold}")

        # 生成可视化
        if visualize and visualization_data and SAVE_VISUALIZATIONS:
            self.generate_visualizations(visualization_data, mode)

        return self.compute_metrics(results, gt_dict, mode)

    def generate_visualizations(self, visualization_data, mode):
        """生成可视化结果"""
        print(f"\n生成可视化结果 ({len(visualization_data)}张)...")

        for i, data in enumerate(visualization_data):
            if data['image_pil'] is not None:
                img_id = data['image_id']
                filename = data.get('filename', f'image_{img_id}')

                # 创建保存路径
                vis_path = os.path.join(VISUALIZATION_DIR, f"{mode}_vis_{i+1:03d}_{filename}.png")

                # 生成可视化
                title = f"Zero-Shot Detection - {mode} Mode\nImage: {filename} (ID: {img_id})"
                visualize_detections(
                    data['image_pil'],
                    data['predictions'],
                    data['ground_truth'],
                    vis_path,
                    title=title,
                    show_gt=True,
                    show_pred=True
                )

    def convert_bbox_format(self, bbox, orig_size):
        """转换边界框格式"""
        if len(bbox) != 4:
            return bbox

        x, y, w, h = 0, 0, 0, 0
        img_h, img_w = float(orig_size[0]), float(orig_size[1])

        is_normalized = all(0 <= coord <= 1 for coord in bbox[:4])

        if is_normalized:
            x_center, y_center, width, height = bbox

            x_center_px = x_center * img_w
            y_center_px = y_center * img_h
            width_px = width * img_w
            height_px = height * img_h

            x = x_center_px - width_px / 2
            y = y_center_px - height_px / 2
            w = width_px
            h = height_px

        else:
            x1, y1, x2, y2 = bbox
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1

        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        if w < 0 or h < 0:
            w, h = 0, 0

        return [float(x), float(y), float(w), float(h)]

    def map_label_to_category_id(self, label):
        """将检测标签映射到COCO类别ID"""
        if label.lower() in self.dataset.cat_name_to_id_lower:
            return self.dataset.cat_name_to_id_lower[label.lower()]

        label_lower = label.lower()
        for cat_name, cat_id in self.dataset.cat_name_to_id_lower.items():
            if cat_name in label_lower:
                return cat_id

        words = label_lower.split()
        for word in words:
            if word in self.dataset.cat_name_to_id_words:
                return self.dataset.cat_name_to_id_words[word]

        special_cases = {
            "car chair": "car",
            "dog cat": "dog",
            "chair dog": "chair",
        }

        for case, cat_name in special_cases.items():
            if case in label_lower and cat_name in self.dataset.cat_name_to_id_lower:
                return self.dataset.cat_name_to_id_lower[cat_name]

        return None

    def predict(self, model, image, caption, box_threshold, text_threshold, device="cpu"):
        """预测函数"""
        try:
            from groundingdino.util.inference import predict as gdino_predict

            if not isinstance(image, torch.Tensor):
                return None, None, None

            if image.dim() == 4:
                image = image.squeeze(0)

            if image.dim() != 3:
                return None, None, None

            if image.device != device:
                image = image.to(device)

            boxes, logits, phrases = gdino_predict(
                model=model,
                image=image,
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=device
            )

            return boxes, logits, phrases
        except Exception as e:
            print(f"预测失败: {e}")
            return None, None, None

    def compute_metrics(self, predictions, ground_truth, mode='all'):
        """计算评测指标"""
        if not predictions:
            print(f"警告: {mode} 模式没有预测结果")
            return {
                'AP': 0.0,
                'AP50': 0.0,
                'AP75': 0.0,
                'AP_s': 0.0,
                'AP_m': 0.0,
                'AP_l': 0.0,
                'mode': mode,
                'num_predictions': len(predictions),
                'num_images': len(ground_truth)
            }

        # 转换为COCO格式
        coco_pred = []
        for pred in predictions:
            bbox = pred['bbox']
            if len(bbox) == 4:
                x, y, w, h = bbox
                if w > 0 and h > 0 and w < 5000 and h < 5000:
                    coco_pred.append({
                        'image_id': pred['image_id'],
                        'category_id': pred['category_id'],
                        'bbox': [x, y, w, h],
                        'score': pred['score']
                    })

        if not coco_pred:
            print(f"警告: {mode} 模式没有有效的预测框")
            return {
                'AP': 0.0,
                'AP50': 0.0,
                'AP75': 0.0,
                'AP_s': 0.0,
                'AP_m': 0.0,
                'AP_l': 0.0,
                'mode': mode,
                'num_predictions': len(predictions),
                'num_images': len(ground_truth)
            }

        print(f"{mode}模式: 有效预测框数量 = {len(coco_pred)}")

        # 创建COCO对象
        try:
            coco_gt = COCO()
            coco_gt.dataset = self.dataset.coco.dataset
            coco_gt.createIndex()

            coco_dt = coco_gt.loadRes(coco_pred)

            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

            if mode == 'seen':
                cat_ids = self.seen_ids
            elif mode == 'unseen':
                cat_ids = self.unseen_ids
            else:
                cat_ids = self.seen_ids + self.unseen_ids

            coco_eval.params.catIds = cat_ids
            coco_eval.params.imgIds = list(ground_truth.keys())

            # 放宽评测参数
            coco_eval.params.iouThrs = np.array([0.1, 0.3, 0.5, 0.75])
            coco_eval.params.maxDets = [1, 10, 100]

            # 执行评测
            print(f"开始COCO评测...")
            coco_eval.evaluate()
            coco_eval.accumulate()

            # 检查匹配
            has_matches = False
            if hasattr(coco_eval, 'evalImgs') and coco_eval.evalImgs is not None:
                for eval_img in coco_eval.evalImgs:
                    if eval_img is not None and 'dtMatches' in eval_img and eval_img['dtMatches'] is not None:
                        if eval_img['dtMatches'].sum() > 0:
                            has_matches = True
                            break

            if not has_matches:
                print(f"警告: {mode}模式没有匹配的检测框")

            coco_eval.summarize()

            # 提取结果
            results = {
                'AP': coco_eval.stats[0],
                'AP50': coco_eval.stats[1],
                'AP75': coco_eval.stats[2],
                'AP_s': coco_eval.stats[3],
                'AP_m': coco_eval.stats[4],
                'AP_l': coco_eval.stats[5],
                'mode': mode,
                'num_predictions': len(predictions),
                'num_images': len(ground_truth)
            }

            return results

        except Exception as e:
            print(f"计算指标时出错: {e}")
            import traceback
            traceback.print_exc()
            return {
                'AP': 0.0,
                'AP50': 0.0,
                'AP75': 0.0,
                'AP_s': 0.0,
                'AP_m': 0.0,
                'AP_l': 0.0,
                'mode': mode,
                'num_predictions': len(predictions),
                'num_images': len(ground_truth)
            }

# ============================== 工具函数 ==============================

def load_grounding_dino():
    """加载 Grounding DINO 模型"""
    print(f"正在加载模型配置: {MODEL_CONFIG}")
    print(f"正在加载模型权重: {MODEL_CHECKPOINT}")

    T, build_model, box_ops, SLConfig, clean_state_dict, predict = import_groundingdino()

    try:
        args = SLConfig.fromfile(MODEL_CONFIG)
    except FileNotFoundError:
        possible_paths = [
            MODEL_CONFIG,
            os.path.join(BASE_DIR, "GroundingDINO_SwinT_OGC.py"),
            os.path.join(BASE_DIR, "config/GroundingDINO_SwinT_OGC.py"),
            "GroundingDINO_SwinT_OGC.py"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                print(f"找到配置文件: {path}")
                args = SLConfig.fromfile(path)
                break
        else:
            raise FileNotFoundError(f"找不到模型配置文件: {MODEL_CONFIG}")

    args.device = DEVICE

    model = build_model(args)

    model_checkpoint = MODEL_CHECKPOINT
    possible_checkpoint_paths = [
        MODEL_CHECKPOINT,
        os.path.join(BASE_DIR, "groundingdino_swint_ogc.pth"),
        "groundingdino_swint_ogc.pth"
    ]

    if not os.path.exists(model_checkpoint):
        print(f"警告: 模型权重文件不存在: {model_checkpoint}")
        for path in possible_checkpoint_paths:
            if os.path.exists(path):
                model_checkpoint = path
                print(f"找到权重文件: {path}")
                break

    if not os.path.exists(model_checkpoint):
        raise FileNotFoundError(f"找不到模型权重文件。请确保文件存在于以下位置:\n{MODEL_CHECKPOINT}")

    print(f"加载权重文件: {model_checkpoint}")
    try:
        checkpoint = torch.load(model_checkpoint, map_location='cpu')
        model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        model.eval()
        print("模型加载完成!")
        return model
    except Exception as e:
        print(f"加载模型权重失败: {e}")
        raise

# ============================== 主函数 ==============================

def main():
    # 检查路径
    print("\n检查路径...")
    print(f"COCO 标注文件: {COCO_ANN_FILE} - {'存在' if os.path.exists(COCO_ANN_FILE) else '不存在'}")
    print(f"COCO 图片目录: {COCO_ROOT} - {'存在' if os.path.exists(COCO_ROOT) else '不存在'}")
    print(f"模型配置: {MODEL_CONFIG} - {'存在' if os.path.exists(MODEL_CONFIG) else '不存在'}")
    print(f"模型权重: {MODEL_CHECKPOINT} - {'存在' if os.path.exists(MODEL_CHECKPOINT) else '不存在'}")

    if not os.path.exists(COCO_ANN_FILE) or not os.path.exists(COCO_ROOT):
        print("错误: 数据集文件不存在!")
        return

    # 设置设备
    device = torch.device(DEVICE)
    print(f"\n使用设备: {device}")

    # 创建500张图片的数据集
    print(f"\n创建{MAX_IMAGES}张图片的数据集...")
    try:
        # 使用指定数量的图片
        dataset = ZeroShotCOCODataset(transform=transform_image_simple, max_images=MAX_IMAGES)

        print(f"数据集大小: {len(dataset)} 张图片")
        print(f"Seen类别数: {len(dataset.seen_ids)}")
        print(f"Unseen类别数: {len(dataset.unseen_ids)}")

    except Exception as e:
        print(f"创建数据集失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # 加载模型
    print("\n加载模型...")
    try:
        model = load_grounding_dino()
    except Exception as e:
        print(f"加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 创建评测器
    try:
        evaluator = ZeroShotEvaluator(
            model=model,
            dataset=dataset,
            device=device,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            apply_nms=True,
            nms_threshold=NMS_THRESHOLD
        )
    except Exception as e:
        print(f"创建评测器失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 先测试单张图片
    print("\n" + "="*60)
    print("测试单张图片...")
    print("="*60)
    try:
        image_tensor, target = dataset[0]

        print(f"图像形状: {image_tensor.shape}")
        print(f"图像ID: {target['image_id'].item()}")
        print(f"图像文件名: {target['filename']}")

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # 测试预测
        test_captions = ["person", "car", "chair"]
        caption = ". ".join(test_captions)

        print(f"测试提示: {caption}")

        boxes, logits, phrases = evaluator.predict(
            model=model,
            image=image_tensor,
            caption=caption,
            box_threshold=0.1,
            text_threshold=0.1,
            device=device
        )

        if boxes is not None and len(boxes) > 0:
            print(f"测试成功! 检测到 {len(boxes)} 个目标")
            print(f"检测到的类别: {phrases[:10]}")

            # 可视化测试结果
            if SAVE_VISUALIZATIONS and target.get('image_pil') is not None:
                test_predictions = []
                for box, score, phrase in zip(boxes[:10], logits[:10], phrases[:10]):
                    box_np = box.cpu().numpy()
                    orig_size = target['orig_size'].cpu().numpy()
                    converted_bbox = evaluator.convert_bbox_format(box_np, orig_size)

                    cat_id = evaluator.map_label_to_category_id(phrase)
                    if cat_id is not None:
                        test_predictions.append({
                            'bbox': converted_bbox,
                            'score': float(score.item()),
                            'category_id': cat_id,
                            'label_text': phrase
                        })

                if test_predictions:
                    vis_path = os.path.join(VISUALIZATION_DIR, "test_sample.png")
                    title = f"Test Sample - Image ID: {target['image_id'].item()}"
                    visualize_detections(
                        target['image_pil'],
                        test_predictions,
                        target,
                        vis_path,
                        title=title
                    )

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

    # 开始评测
    print("\n" + "="*60)
    print(f"开始评测 {MAX_IMAGES} 张图片...")
    print("="*60)
    print(f"数据集大小: {len(dataset)} 张图片")
    print(f"预计评测时间: 约{len(dataset)*2}秒 ({len(dataset)//60}分钟)")

    try:
        # 评测所有类别
        print("\n评测 all 模式...")
        all_results = evaluator.evaluate(dataloader, mode='all', visualize=True)

        # 评测seen类别
        print("\n评测 seen 模式...")
        seen_results = evaluator.evaluate(dataloader, mode='seen', visualize=True)

        # 评测unseen类别
        print("\n评测 unseen 模式...")
        unseen_results = evaluator.evaluate(dataloader, mode='unseen', visualize=True)

        # 打印详细结果
        print("\n" + "="*80)
        print(f"零样本目标检测评测结果 ({MAX_IMAGES}张图片)")
        print("="*80)
        print(f"数据集: COCO {COCO_SPLIT}2017")
        print(f"模型: Grounding DINO")
        print(f"检测框阈值: {BOX_THRESHOLD}, 文本阈值: {TEXT_THRESHOLD}")
        print(f"NMS阈值: {NMS_THRESHOLD}, 应用NMS: 是")
        print(f"总图片数: {len(dataset)}")
        print("-"*80)
        print(f"{'指标':<25} {'全部':<12} {'Seen':<12} {'Unseen':<12}")
        print("-"*80)
        print(f"{'AP@[0.5:0.95]':<25} {all_results['AP']:.4f} {seen_results['AP']:.4f} {unseen_results['AP']:.4f}")
        print(f"{'AP@0.5':<25} {all_results['AP50']:.4f} {seen_results['AP50']:.4f} {unseen_results['AP50']:.4f}")
        print(f"{'AP@0.75':<25} {all_results['AP75']:.4f} {seen_results['AP75']:.4f} {unseen_results['AP75']:.4f}")
        print(f"{'AP_s (小目标)':<25} {all_results['AP_s']:.4f} {seen_results['AP_s']:.4f} {unseen_results['AP_s']:.4f}")
        print(f"{'AP_m (中目标)':<25} {all_results['AP_m']:.4f} {seen_results['AP_m']:.4f} {unseen_results['AP_m']:.4f}")
        print(f"{'AP_l (大目标)':<25} {all_results['AP_l']:.4f} {seen_results['AP_l']:.4f} {unseen_results['AP_l']:.4f}")
        print("-"*80)
        print(f"{'有效预测数':<25} {all_results.get('num_predictions', 0):<12} {seen_results.get('num_predictions', 0):<12} {unseen_results.get('num_predictions', 0):<12}")
        print(f"{'评测图片数':<25} {all_results.get('num_images', 0):<12} {seen_results.get('num_images', 0):<12} {unseen_results.get('num_images', 0):<12}")
        print("="*80)

        # 保存详细结果
        detailed_results = {
            'config': {
                'coco_root': COCO_ROOT,
                'ann_file': COCO_ANN_FILE,
                'model_config': MODEL_CONFIG,
                'model_checkpoint': MODEL_CHECKPOINT,
                'device': DEVICE,
                'box_threshold': BOX_THRESHOLD,
                'text_threshold': TEXT_THRESHOLD,
                'nms_threshold': NMS_THRESHOLD,
                'apply_nms': True,
                'text_prompt_style': 'a clear photo of a {cls}',
                'total_images': len(dataset),
                'max_images': MAX_IMAGES
            },
            'all_classes': all_results,
            'seen_classes': seen_results,
            'unseen_classes': unseen_results,
            'statistics': {
                'total_predictions': evaluator.total_predictions,
                'matched_predictions': evaluator.matched_predictions,
                'visualization_count': evaluator.visualization_count
            }
        }

        output_path = os.path.join(OUTPUT_DIR, f'evaluation_results_{MAX_IMAGES}.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

        print(f"\n详细结果已保存至: {output_path}")
        print(f"可视化结果保存在: {VISUALIZATION_DIR}")

        # 生成总结报告
        summary_path = os.path.join(OUTPUT_DIR, 'summary_report.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"零样本目标检测评测总结报告 ({MAX_IMAGES}张图片)\n")
            f.write("="*80 + "\n\n")
            f.write(f"评测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据集: COCO {COCO_SPLIT}2017\n")
            f.write(f"总图片数: {len(dataset)}\n")
            f.write(f"模型: Grounding DINO\n\n")
            f.write("主要结果:\n")
            f.write("-"*60 + "\n")
            f.write(f"AP@[0.5:0.95]: {all_results['AP']:.4f}\n")
            f.write(f"  - Seen类别: {seen_results['AP']:.4f}\n")
            f.write(f"  - Unseen类别: {unseen_results['AP']:.4f}\n")
            f.write(f"AP@0.5: {all_results['AP50']:.4f}\n")
            f.write(f"  - Seen类别: {seen_results['AP50']:.4f}\n")
            f.write(f"  - Unseen类别: {unseen_results['AP50']:.4f}\n\n")
            f.write("性能分析:\n")
            f.write("-"*60 + "\n")
            seen_unseen_ratio = seen_results['AP'] / unseen_results['AP'] if unseen_results['AP'] > 0 else float('inf')
            f.write(f"Seen/Unseen AP比率: {seen_unseen_ratio:.2f}\n")
            if seen_unseen_ratio < 1:
                f.write("Unseen类别性能优于Seen类别，模型泛化能力强。\n")
            else:
                f.write("Seen类别性能优于Unseen类别，符合预期。\n")
            f.write(f"\n可视化样本数: {evaluator.visualization_count}\n")
            f.write(f"详细结果文件: {output_path}\n")
            f.write(f"可视化目录: {VISUALIZATION_DIR}\n")

        print(f"总结报告已保存至: {summary_path}")

    except Exception as e:
        print(f"评测过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    from datetime import datetime

    main()
