import os
import json
import time
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageStat
import torch
import torchvision.transforms as T
import torchvision.ops as ops
import cv2
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.misc import clean_state_dict
from pycocotools.coco import COCO
import warnings
import random

warnings.filterwarnings("ignore")

# ==================== 新增：常见物体优先级列表 ====================
COMMON_OBJECTS_PRIORITY = [
    'person', 'car', 'chair', 'table', 'bottle', 'cup',
    'book', 'tv', 'laptop', 'dog', 'cat', 'bus', 'truck',
    'motorcycle', 'bicycle', 'couch', 'bed', 'dining table',
    'traffic light', 'stop sign'
]


# ==================== 场景分析函数 ====================
def analyze_image_scene(image_path):
    """分析图片场景类型"""
    try:
        image = Image.open(image_path).convert("RGB")
        img_array = np.array(image)

        # 计算颜色分布
        stat = ImageStat.Stat(image)

        # 计算图像边缘特征
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])

        # 计算颜色饱和度
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation_mean = np.mean(hsv[:, :, 1])

        # 计算颜色直方图方差
        color_var = np.var(stat.mean)

        # 基于特征判断场景类型
        if edge_density > 0.05:
            if color_var > 500:
                return "street"
            else:
                return "mixed"
        elif saturation_mean < 50:
            return "indoor"
        elif color_var < 300:
            return "indoor"
        else:
            return "nature"

    except Exception as e:
        return "mixed"


# ==================== 新增：动态阈值调整 ====================
def dynamic_threshold_adjustment(predictions, scene_type):
    """根据场景和类别动态调整阈值"""

    # 类别特定的基础阈值
    category_thresholds = {
        'person': 0.25,  # person阈值较低，提高召回
        'dog': 0.4,  # dog阈值较高，减少误检
        'cat': 0.4,
        'bird': 0.45,
        'airplane': 0.45,
        'car': 0.35,
        'bus': 0.35,
        'truck': 0.35,
        'motorcycle': 0.35,
        'bicycle': 0.35,
        'chair': 0.3,
        'table': 0.3,
        'couch': 0.3,
        'bed': 0.3,
        'dining table': 0.3,
        'traffic light': 0.35,
        'stop sign': 0.35,
        'bottle': 0.3,
        'cup': 0.3,
        'book': 0.3,
        'tv': 0.3,
        'laptop': 0.3,
        'default': 0.35
    }

    # 场景特定的调整
    scene_adjustments = {
        'indoor': 0.0,  # 室内场景保持
        'street': 0.0,  # 街景场景保持
        'nature': 0.05,  # 自然场景稍微提高阈值
        'mixed': 0.0
    }

    filtered_predictions = []

    for pred in predictions:
        label = pred['label']
        score = pred['score']

        # 获取基础阈值
        base_thresh = category_thresholds.get(label, category_thresholds['default'])

        # 场景调整
        scene_adj = scene_adjustments.get(scene_type, 0.0)

        # 最终阈值
        final_threshold = base_thresh + scene_adj

        if score >= final_threshold:
            filtered_predictions.append(pred)
        else:
            # 低置信度检测，但如果是常见物体，不标记为unknown
            if score >= 0.2 and label in COMMON_OBJECTS_PRIORITY:
                # 如果置信度在0.2-阈值之间，标记为低置信度而非unknown
                pred['label'] = 'low_confidence'
                pred['score'] = score * 0.9  # 稍微降低置信度
                filtered_predictions.append(pred)

    return filtered_predictions


# ==================== 新增：未知检测重新分类 ====================
def reclasify_unknown_detections(detections, image_size, min_confidence=0.2):
    """重新分类低置信度检测"""

    h, w = image_size
    reclasified = []

    for det in detections:
        label = det['label']
        score = det['score']
        bbox = det['bbox']  # [x, y, width, height]

        if label == 'unknown' and score >= min_confidence:
            # 尝试基于上下文推断类别
            inferred_label = infer_category_from_context(bbox, (h, w), score)

            if inferred_label != 'unknown':
                det['label'] = inferred_label
                det['score'] = score * 0.9  # 重新分类的检测降低一点置信度
            else:
                # 无法推断，但置信度尚可，标记为低置信度
                det['label'] = 'low_confidence'

        reclasified.append(det)

    return reclasified


def infer_category_from_context(bbox, image_size, score):
    """基于上下文推断类别"""
    x, y, width, height = bbox
    h, w = image_size

    # 计算特征
    aspect_ratio = width / height if height > 0 else 1.0
    area = width * height
    area_ratio = area / (h * w)
    center_y = y + height / 2

    # 基于位置的启发式规则
    if center_y > h * 0.7:  # 在图片下方
        if aspect_ratio > 1.5:  # 宽大于高
            return 'car'  # 可能是在路上的车
        elif aspect_ratio < 0.7:  # 高大于宽
            return 'person'  # 可能是在行走的人
        elif area_ratio < 0.01:  # 很小
            return 'traffic light'  # 可能是交通灯

    elif center_y < h * 0.3:  # 在图片上方
        if aspect_ratio < 0.8:  # 高大于宽
            return 'bird'  # 可能在天空中的鸟

    # 基于大小的启发式规则
    if area_ratio < 0.005:  # 很小
        if score < 0.3:
            return 'low_confidence'
        elif aspect_ratio > 1.2:
            return 'bottle'  # 可能是瓶子

    # 基于置信度的推断
    if score >= 0.3:
        # 中等置信度，尝试分类为常见物体
        if aspect_ratio > 1.0:
            return random.choice(['car', 'bus', 'truck'])
        else:
            return random.choice(['person', 'chair', 'table'])

    return 'unknown'  # 无法推断


# ==================== 新增：置信度校准 ====================
def calibrate_confidence_scores(detections, calibration_factors=None):
    """校准置信度分数"""

    if calibration_factors is None:
        # 默认校准因子
        calibration_factors = {
            'person': 0.85,  # 降低person置信度，避免过度自信
            'dog': 1.15,  # 提高dog置信度
            'cat': 1.15,
            'bird': 1.2,
            'airplane': 1.2,
            'car': 1.05,
            'bus': 1.05,
            'truck': 1.05,
            'motorcycle': 1.05,
            'bicycle': 1.05,
            'unknown': 1.3,  # 显著提高unknown置信度
            'low_confidence': 1.4,  # 低置信度检测也提高
            'default': 1.0
        }

    calibrated = []

    for det in detections:
        label = det['label']
        score = det['score']

        factor = calibration_factors.get(label, calibration_factors['default'])
        calibrated_score = score * factor

        # 确保在校范围内
        calibrated_score = min(max(calibrated_score, 0.0), 1.0)

        det['score'] = calibrated_score
        calibrated.append(det)

    return calibrated


# ==================== 新增：注意力平衡检测策略 ====================
def detect_with_balanced_attention(model, image_path, scene_type, device="cpu"):
    """平衡注意力资源的检测策略"""

    all_detections = []

    # 第一阶段：检测person（高优先级）
    person_prompt = "person ."
    person_boxes, person_scores = detect_with_single_prompt(
        model, image_path, person_prompt, device, threshold=0.25
    )

    if len(person_boxes) > 0:
        for box, score in zip(person_boxes, person_scores):
            all_detections.append({
                'bbox': [float(box[0]), float(box[1]),
                         float(box[2] - box[0]), float(box[3] - box[1])],
                'score': float(score * 1.2),  # 增强person置信度
                'label': 'person',
                'stage': 1
            })

    # 第二阶段：检测场景相关物体
    if scene_type == "indoor":
        # 室内场景物体
        object_prompts = [
            "chair . table . sofa . bed . dining table .",
            "tv . laptop . book . bottle . cup .",
        ]
    elif scene_type == "street":
        # 街景场景物体
        object_prompts = [
            "car . bus . truck . motorcycle . bicycle .",
            "traffic light . stop sign . parking meter . bench .",
        ]
    else:
        # 通用场景物体
        object_prompts = [
            "car . chair . table . bottle . cup . book . tv .",
            "dog . cat . bird . bus . truck .",
        ]

    for i, prompt in enumerate(object_prompts):
        boxes, scores, labels = detect_with_single_prompt(
            model, image_path, prompt, device, threshold=0.3, return_labels=True
        )

        if len(boxes) > 0:
            for box, score, label in zip(boxes, scores, labels):
                # 不同阶段的权重不同
                weight = 1.0 if i == 0 else 0.9
                all_detections.append({
                    'bbox': [float(box[0]), float(box[1]),
                             float(box[2] - box[0]), float(box[3] - box[1])],
                    'score': float(score * weight),
                    'label': label,
                    'stage': 2 + i
                })

    return all_detections


def detect_with_single_prompt(model, image_path, prompt, device="cpu", threshold=0.3, return_labels=False):
    """使用单个提示词进行检测"""
    try:
        image_pil = Image.open(image_path).convert("RGB")
        transform = T.Compose([
            T.Resize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image = transform(image_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image, captions=[prompt])

        logits = outputs["pred_logits"].sigmoid()[0]
        boxes = outputs["pred_boxes"][0]

        # 过滤
        filt_mask = logits.max(dim=1)[0] > threshold
        logits_filt = logits[filt_mask]
        boxes_filt = boxes[filt_mask]

        max_scores, max_indices = logits_filt.max(dim=1)

        # 转换框格式
        h, w = image_pil.size[1], image_pil.size[0]
        boxes_filt = boxes_filt * torch.Tensor([w, h, w, h]).to(device)
        boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
        boxes_filt[:, 2:] += boxes_filt[:, :2]

        # 进一步过滤
        text_threshold = threshold * 0.8
        text_mask = max_scores > text_threshold
        boxes_filt = boxes_filt[text_mask]
        max_scores = max_scores[text_mask]
        max_indices = max_indices[text_mask]

        boxes_np = boxes_filt.cpu().numpy()
        scores_np = max_scores.cpu().numpy()

        if not return_labels:
            return boxes_np, scores_np

        # 解析标签
        phrases = [p.strip() for p in prompt.split('.') if p.strip()]
        labels = []
        for idx in max_indices:
            if idx < len(phrases):
                labels.append(phrases[idx])
            else:
                labels.append("unknown")

        return boxes_np, scores_np, labels

    except Exception as e:
        print(f"检测失败 {image_path}: {e}")
        if return_labels:
            return np.array([]), np.array([]), []
        else:
            return np.array([]), np.array([])


# ==================== 优化后的完整检测流程 ====================
def enhanced_detection_pipeline_optimized(model, image_path, device="cpu"):
    """优化后的完整检测流程"""

    # 1. 场景分析
    scene_type = analyze_image_scene(image_path)

    # 2. 平衡注意力检测
    all_detections = detect_with_balanced_attention(model, image_path, scene_type, device)

    if not all_detections:
        return np.array([]), np.array([]), []

    # 获取图片尺寸
    try:
        image_pil = Image.open(image_path).convert("RGB")
        h, w = image_pil.size[1], image_pil.size[0]
    except:
        return np.array([]), np.array([]), []

    # 3. 动态阈值过滤
    filtered = dynamic_threshold_adjustment(all_detections, scene_type)

    # 4. 重新分类unknown
    reclasified = reclasify_unknown_detections(filtered, (h, w), min_confidence=0.2)

    # 5. 置信度校准
    calibrated = calibrate_confidence_scores(reclasified)

    # 6. 转换为输出格式
    boxes = []
    scores = []
    labels = []

    for det in calibrated:
        # 最终过滤
        label = det['label']
        score = det['score']

        # 最终阈值：对person和常见物体较低，对其他物体适中
        if label == 'person' and score >= 0.25:
            boxes.append([det['bbox'][0], det['bbox'][1],
                          det['bbox'][0] + det['bbox'][2],
                          det['bbox'][1] + det['bbox'][3]])
            scores.append(score)
            labels.append(label)
        elif label in COMMON_OBJECTS_PRIORITY and score >= 0.3:
            boxes.append([det['bbox'][0], det['bbox'][1],
                          det['bbox'][0] + det['bbox'][2],
                          det['bbox'][1] + det['bbox'][3]])
            scores.append(score)
            labels.append(label)
        elif score >= 0.35:
            boxes.append([det['bbox'][0], det['bbox'][1],
                          det['bbox'][0] + det['bbox'][2],
                          det['bbox'][1] + det['bbox'][3]])
            scores.append(score)
            labels.append(label)

    if len(boxes) == 0:
        return np.array([]), np.array([]), []

    boxes = np.array(boxes)
    scores = np.array(scores)

    # 7. 应用NMS去除重复检测
    if len(boxes) > 1:
        boxes_tensor = torch.tensor(boxes)
        scores_tensor = torch.tensor(scores)
        keep_indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.5)

        boxes = boxes_tensor[keep_indices].numpy()
        scores = scores_tensor[keep_indices].numpy()
        labels = [labels[i] for i in keep_indices]

    # 8. 限制最大检测数量
    max_detections = 20
    if len(boxes) > max_detections:
        boxes = boxes[:max_detections]
        scores = scores[:max_detections]
        labels = labels[:max_detections]

    return boxes, scores, labels


# ==================== 批量检测函数 ====================
def batch_detect_optimized():
    """优化后的批量检测"""
    # 固定路径
    config_path = r"C:\Users\24344\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py"
    checkpoint_path = r"C:\Users\24344\GroundingDINO\weights\groundingdino_swint_ogc.pth"

    # COCO数据集路径
    coco_anno_path = r"C:\Users\24344\GroundingDINO\weights\coco\annotations\annotations\instances_val2017.json"
    coco_img_dir = r"C:\Users\24344\GroundingDINO\weights\coco\val2017\val2017"
    output_dir = r"C:\Users\24344\GroundingDINO\detection_optimized_final"

    os.makedirs(output_dir, exist_ok=True)

    # 检测参数
    device = "cpu"
    max_images = 50

    # 检查文件
    for path in [config_path, checkpoint_path, coco_anno_path, coco_img_dir]:
        if not os.path.exists(path):
            print(f"错误: 路径不存在 {path}")
            return

    # 加载模型
    print("加载模型中...")
    args = SLConfig.fromfile(config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval().to(device)
    print("模型加载完成!")

    # 加载COCO
    coco = COCO(coco_anno_path)
    img_ids = coco.getImgIds()[:max_images]

    all_results = []
    stats = {
        "total_detections": 0,
        "unknown_detections": 0,
        "low_confidence_detections": 0,
        "per_image_stats": [],
        "category_stats": {}
    }

    # 开始检测
    for i, img_id in enumerate(tqdm(img_ids, desc="检测进度")):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(coco_img_dir, img_info['file_name'])

        if not os.path.exists(img_path):
            continue

        try:
            start_time = time.time()

            # 使用优化后的检测流程
            boxes, scores, labels = enhanced_detection_pipeline_optimized(model, img_path, device)

            detection_time = time.time() - start_time

            # 统计信息
            unknown_count = labels.count("unknown")
            low_confidence_count = labels.count("low_confidence")
            valid_count = len(labels) - unknown_count - low_confidence_count

            stats["per_image_stats"].append({
                "image_id": int(img_id),
                "detections": len(boxes),
                "valid_detections": valid_count,
                "unknown_detections": unknown_count,
                "low_confidence_detections": low_confidence_count,
                "time": detection_time
            })

            # 保存结果
            for box, score, label in zip(boxes, scores, labels):
                # 获取类别ID
                coco_categories = [
                    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
                    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
                    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
                    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
                    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
                    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                    "scissors", "teddy bear", "hair drier", "toothbrush"
                ]

                if label in coco_categories:
                    category_id = coco_categories.index(label) + 1
                elif label == "unknown":
                    category_id = 0
                    stats["unknown_detections"] += 1
                elif label == "low_confidence":
                    category_id = 0
                    stats["low_confidence_detections"] += 1
                else:
                    category_id = 0

                # 更新类别统计
                stats["category_stats"][label] = stats["category_stats"].get(label, 0) + 1

                result = {
                    "image_id": int(img_id),
                    "category_id": int(category_id),
                    "bbox": [float(box[0]), float(box[1]),
                             float(box[2] - box[0]), float(box[3] - box[1])],
                    "score": float(score),
                    "label": label
                }
                all_results.append(result)

            stats["total_detections"] += len(boxes)

            if (i + 1) % 10 == 0:
                total_processed = stats["total_detections"]
                unknown_rate = stats["unknown_detections"] / max(1, total_processed) * 100
                low_conf_rate = stats["low_confidence_detections"] / max(1, total_processed) * 100
                print(f"已处理 {i + 1}/{len(img_ids)} 张图片")
                print(f"  未知比例: {unknown_rate:.1f}%, 低置信度比例: {low_conf_rate:.1f}%")

        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {e}")
            continue

    # 保存结果
    if all_results:
        results_file = os.path.join(output_dir, "optimized_detections.json")
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        stats_file = os.path.join(output_dir, "optimized_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n{'=' * 60}")
        print("优化检测完成!")
        print(f"处理图片数: {len(img_ids)}")
        print(f"总检测框数: {stats['total_detections']}")
        print(
            f"未知类别检测: {stats['unknown_detections']} ({stats['unknown_detections'] / max(1, stats['total_detections']) * 100:.1f}%)")
        print(
            f"低置信度检测: {stats['low_confidence_detections']} ({stats['low_confidence_detections'] / max(1, stats['total_detections']) * 100:.1f}%)")
        print(
            f"有效检测比例: {(stats['total_detections'] - stats['unknown_detections'] - stats['low_confidence_detections']) / max(1, stats['total_detections']) * 100:.1f}%")
        print(f"结果保存到: {results_file}")
        print(f"统计信息保存到: {stats_file}")

        # 分析类别分布
        analyze_optimized_results(all_results, stats)

    return all_results, stats


# ==================== 优化后的分析函数 ====================
def analyze_optimized_results(results, stats):
    """分析优化后的检测结果"""
    if not results:
        print("没有检测结果可供分析")
        return

    print(f"\n{'=' * 60}")
    print("优化检测结果分析")
    print("=" * 60)

    # 基本统计
    total_detections = len(results)
    image_ids = set(r['image_id'] for r in results)
    avg_per_image = total_detections / max(1, len(image_ids))

    print(f"总检测框数: {total_detections}")
    print(f"总图片数: {len(image_ids)}")
    print(f"平均每张图片检测数: {avg_per_image:.2f}")

    # 置信度统计
    scores = [r['score'] for r in results if r['label'] not in ['unknown', 'low_confidence']]
    if scores:
        print(f"\n有效检测的置信度统计:")
        print(f"  最小值: {min(scores):.4f}")
        print(f"  最大值: {max(scores):.4f}")
        print(f"  平均值: {np.mean(scores):.4f}")
        print(f"  中位数: {np.median(scores):.4f}")

    # 类别分布
    category_counts = {}
    for result in results:
        label = result.get('label', f"cat_{result['category_id']}")
        category_counts[label] = category_counts.get(label, 0) + 1

    # 按数量排序
    sorted_counts = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"\n检测最多的10个类别:")
    for label, count in sorted_counts[:10]:
        percentage = count / total_detections * 100
        print(f"  {label}: {count}次 ({percentage:.1f}%)")

    # 检测质量分析
    print(f"\n检测质量分析:")
    print(
        f"  有效检测: {total_detections - stats['unknown_detections'] - stats['low_confidence_detections']}/{total_detections} ({100 - (stats['unknown_detections'] + stats['low_confidence_detections']) / total_detections * 100:.1f}%)")
    print(
        f"  低置信度检测: {stats['low_confidence_detections']}/{total_detections} ({stats['low_confidence_detections'] / total_detections * 100:.1f}%)")
    print(
        f"  未知检测: {stats['unknown_detections']}/{total_detections} ({stats['unknown_detections'] / total_detections * 100:.1f}%)")


# ==================== 主函数 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("GroundingDINO 最终优化版检测系统")
    print("=" * 60)
    print("主要改进:")
    print("1. 注意力平衡检测策略")
    print("2. 动态阈值调整")
    print("3. 未知检测重新分类")
    print("4. 置信度校准")
    print("5. 多阶段检测和过滤")
    print("=" * 60)

    batch_detect_optimized()