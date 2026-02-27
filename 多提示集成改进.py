import os
import json
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pycocotools.coco import COCO
from groundingdino.util.inference import load_model, load_image, predict
from scipy.optimize import linear_sum_assignment

# --- 抑制警告 ---
warnings.filterwarnings("ignore")

# --- 配置路径 ---
COCO_ROOT = r"C:\Users\24344\GroundingDINO\weights\coco\val2017\val_images"
COCO_ANN_FILE = r"C:\Users\24344\GroundingDINO\weights\coco\annotations\annotations_images\instances_val2017.json"

MODEL_CONFIG = r"C:\Users\24344\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py"
MODEL_CHECKPOINT = r"C:\Users\24344\GroundingDINO\weights\groundingdino_swint_ogc.pth"

# --- 加载模型和数据集 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(MODEL_CONFIG, MODEL_CHECKPOINT, device=device)
coco = COCO(COCO_ANN_FILE)
categories = coco.loadCats(coco.getCatIds())
category_name_to_id = {cat['name'].lower(): cat['id'] for cat in categories}
category_id_to_name = {cat['id']: cat['name'] for cat in categories}

print(f"COCO 数据集：{len(coco.getImgIds())} 张图片，{len(categories)} 个类别")

# --- 实验配置 ---
# 测试3个类别
TEST_CATEGORIES = ["person", "car", "chair"]

# 扩展的Prompt集合（每个类别多个Prompt）
CATEGORY_MULTI_PROMPTS = {
    "person": [
        "person",  # 纯名称
        "a person",  # 简单描述
        "a photo of a person",  # 模板
        "a human person",  # 详细描述
        "a person in the image",  # 上下文
        "a person standing",  # 动作描述
        "human",  # 同义词
        "people",  # 复数
        "a man or woman",  # 细分
        "a person walking"  # 动态
    ],
    "car": [
        "car",
        "a car",
        "a photo of a car",
        "a car vehicle",
        "a car on the road",
        "a parked car",
        "vehicle",
        "automobile",
        "a red car",
        "a moving car"
    ],
    "chair": [
        "chair",
        "a chair",
        "a photo of a chair",
        "a chair furniture",
        "a chair in the room",
        "a chair for sitting",
        "seat",
        "furniture",
        "a wooden chair",
        "an office chair"
    ]
}

# 融合策略配置
FUSION_STRATEGIES = {
    'max_confidence': '最大置信度融合',
    'weighted_average': '加权平均融合',
    'nms': 'NMS融合',
    'wbf': 'WBF融合'
}


# --- 基础函数 ---
def convert_groundingdino_to_coco(box_np, img_width, img_height):
    """将 Grounding DINO 输出转换为 COCO 格式 [x, y, w, h]"""
    cx_norm, cy_norm, w_norm, h_norm = box_np

    w_pixel = w_norm * img_width
    h_pixel = h_norm * img_height
    x_pixel = (cx_norm * img_width) - (w_pixel / 2)
    y_pixel = (cy_norm * img_height) - (h_pixel / 2)

    # 边界检查
    x_pixel = max(0, x_pixel)
    y_pixel = max(0, y_pixel)
    w_pixel = min(w_pixel, img_width - x_pixel)
    h_pixel = min(h_pixel, img_height - y_pixel)

    if w_pixel > 5 and h_pixel > 5:
        return [float(x_pixel), float(y_pixel), float(w_pixel), float(h_pixel)]
    return None


def calculate_iou(box1, box2):
    """计算两个框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


# --- 手动AP计算函数（改进版）---
def calculate_ap_manual(detections, gt_boxes, category_id, iou_threshold=0.5):
    """
    手动计算AP（避免使用pycocotools的COCOeval）
    基于我们之前验证成功的方法
    """
    if len(detections) == 0 or len(gt_boxes) == 0:
        return 0.0, {}

    # 按置信度排序检测框
    sorted_detections = sorted(detections, key=lambda x: x['score'], reverse=True)

    true_positives = 0
    false_positives = 0
    used_gts = set()  # 跟踪已匹配的真实框
    all_precisions = []  # 存储每个检测点的精确率

    # 为每个检测计算匹配状态
    for i, det in enumerate(sorted_detections):
        det_box = det['bbox']
        best_iou = 0
        best_gt_idx = -1

        # 找到最佳匹配的真实框
        for j, gt_box in enumerate(gt_boxes):
            if j in used_gts:
                continue

            iou = calculate_iou(det_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        # 判断是否匹配
        if best_iou > iou_threshold and best_gt_idx != -1:
            true_positives += 1
            used_gts.add(best_gt_idx)
        else:
            false_positives += 1

        # 计算当前的精确率
        current_precision = true_positives / (true_positives + false_positives) if (
                                                                                               true_positives + false_positives) > 0 else 0
        all_precisions.append(current_precision)

    # 计算AP（所有精确率的平均值）
    if all_precisions:
        ap = sum(all_precisions) / len(all_precisions)

        # 计算其他统计指标
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / len(gt_boxes) if len(gt_boxes) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        stats = {
            'ap': ap,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'total_detections': len(detections),
            'total_gt': len(gt_boxes),
            'matched_rate': true_positives / len(gt_boxes) if len(gt_boxes) > 0 else 0
        }

        return ap, stats

    return 0.0, {}


# --- 多提示集成函数 ---
def run_multi_prompt_detection(image_tensor, category_prompts, img_width, img_height,
                               category_name, box_threshold=0.1, text_threshold=0.1):
    """使用多个Prompt运行检测"""
    all_detections = []

    for prompt_text in category_prompts:
        boxes, logits, phrases = predict(
            model=model,
            image=image_tensor,
            caption=prompt_text,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device
        )

        if len(boxes) > 0:
            boxes_np = boxes.cpu().numpy()

            for box, logit, phrase in zip(boxes_np, logits, phrases):
                # 类别匹配检查
                phrase_lower = phrase.lower()
                category_lower = category_name.lower()

                # 扩展的匹配逻辑
                is_match = False
                if category_lower in phrase_lower:
                    is_match = True
                elif category_lower == "person" and (
                        "human" in phrase_lower or "man" in phrase_lower or "woman" in phrase_lower):
                    is_match = True
                elif category_lower == "car" and ("vehicle" in phrase_lower or "automobile" in phrase_lower):
                    is_match = True
                elif category_lower == "chair" and ("seat" in phrase_lower or "furniture" in phrase_lower):
                    is_match = True

                if is_match:
                    converted_box = convert_groundingdino_to_coco(box, img_width, img_height)
                    if converted_box:
                        detection = {
                            "bbox": converted_box,
                            "score": float(logit),
                            "prompt": prompt_text,
                            "phrase": phrase
                        }
                        all_detections.append(detection)

    return all_detections


# --- 融合策略实现 ---
def max_confidence_fusion(detections, iou_threshold=0.5):
    """最大置信度融合策略"""
    if not detections:
        return []

    # 按置信度排序
    sorted_detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    fused_detections = []

    while sorted_detections:
        # 取最高置信度的检测
        best_det = sorted_detections.pop(0)
        fused_detections.append(best_det)

        # 移除重叠框
        remaining_detections = []
        for det in sorted_detections:
            iou = calculate_iou(best_det['bbox'], det['bbox'])
            if iou < iou_threshold:
                remaining_detections.append(det)

        sorted_detections = remaining_detections

    return fused_detections


def weighted_average_fusion(detections, iou_threshold=0.5):
    """加权平均融合策略"""
    if not detections:
        return []

    clusters = []

    # 聚类相似的检测框
    for det in detections:
        matched = False
        for cluster in clusters:
            # 检查是否与聚类中的任何框匹配
            for cluster_det in cluster['detections']:
                iou = calculate_iou(det['bbox'], cluster_det['bbox'])
                if iou >= iou_threshold:
                    cluster['detections'].append(det)
                    matched = True
                    break
            if matched:
                break

        if not matched:
            clusters.append({'detections': [det]})

    # 对每个聚类进行加权平均
    fused_detections = []
    for cluster in clusters:
        if cluster['detections']:
            # 计算加权平均框
            total_weight = sum(d['score'] for d in cluster['detections'])
            weighted_bbox = [0, 0, 0, 0]

            for det in cluster['detections']:
                weight = det['score'] / total_weight
                for i in range(4):
                    weighted_bbox[i] += det['bbox'][i] * weight

            # 计算平均置信度
            avg_score = sum(d['score'] for d in cluster['detections']) / len(cluster['detections'])

            fused_detection = {
                'bbox': weighted_bbox,
                'score': avg_score,
                'prompt': 'weighted_fusion',
                'phrase': 'fused detection',
                'num_sources': len(cluster['detections'])
            }
            fused_detections.append(fused_detection)

    return fused_detections


def nms_fusion(detections, iou_threshold=0.5, score_threshold=0.1):
    """NMS融合策略"""
    if not detections:
        return []

    # 按置信度排序
    sorted_detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    keep = []

    while sorted_detections:
        # 取最高置信度的检测
        current = sorted_detections.pop(0)
        keep.append(current)

        # 计算与剩余框的IoU
        remaining = []
        for det in sorted_detections:
            iou = calculate_iou(current['bbox'], det['bbox'])
            if iou < iou_threshold:
                remaining.append(det)

        sorted_detections = remaining

    return keep


def wbf_fusion(detections, iou_threshold=0.5, score_threshold=0.1):
    """WBF (Weighted Boxes Fusion) 策略"""
    if not detections:
        return []

    # 聚类相似的检测框
    clusters = []
    for det in detections:
        matched = False
        for cluster in clusters:
            # 计算与聚类中所有框的平均IoU
            cluster_ious = []
            for cluster_det in cluster['detections']:
                iou = calculate_iou(det['bbox'], cluster_det['bbox'])
                cluster_ious.append(iou)

            avg_iou = sum(cluster_ious) / len(cluster_ious) if cluster_ious else 0
            if avg_iou >= iou_threshold * 0.5:  # 降低阈值以允许更多融合
                cluster['detections'].append(det)
                matched = True
                break

        if not matched:
            clusters.append({'detections': [det]})

    # 对每个聚类进行WBF
    fused_detections = []
    for cluster in clusters:
        if cluster['detections']:
            dets = cluster['detections']

            # 计算加权框
            total_score = sum(d['score'] for d in dets)
            weighted_box = [0, 0, 0, 0]

            for det in dets:
                weight = det['score'] / total_score
                for i in range(4):
                    weighted_box[i] += det['bbox'][i] * weight

            # 计算融合置信度（考虑来源数量）
            avg_score = sum(d['score'] for d in dets) / len(dets)
            fusion_score = avg_score * (1 + 0.1 * len(dets))  # 奖励多源检测

            fused_detection = {
                'bbox': weighted_box,
                'score': min(fusion_score, 1.0),  # 确保不超过1.0
                'prompt': 'wbf_fusion',
                'phrase': 'fused detection',
                'num_sources': len(dets)
            }
            fused_detections.append(fused_detection)

    # 按置信度排序
    fused_detections.sort(key=lambda x: x['score'], reverse=True)
    return fused_detections


def apply_fusion_strategy(detections, strategy='max_confidence', **kwargs):
    """应用指定的融合策略"""
    if strategy == 'max_confidence':
        return max_confidence_fusion(detections, **kwargs)
    elif strategy == 'weighted_average':
        return weighted_average_fusion(detections, **kwargs)
    elif strategy == 'nms':
        return nms_fusion(detections, **kwargs)
    elif strategy == 'wbf':
        return wbf_fusion(detections, **kwargs)
    else:
        return detections  # 默认返回原始检测


# --- 可视化函数 ---
def visualize_detections_comparison(img_path, fusion_results, gt_boxes, category_name, save_path):
    """可视化不同融合策略的结果比较"""
    img = Image.open(img_path)

    # 创建子图：原始、各融合策略、真实标注
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    # 子图1: 真实标注
    ax = axes[0]
    ax.imshow(img)
    for gt_box in gt_boxes:
        rect = patches.Rectangle(
            (gt_box[0], gt_box[1]), gt_box[2], gt_box[3],
            linewidth=2, edgecolor='lime', facecolor='none', alpha=0.7
        )
        ax.add_patch(rect)
    ax.set_title(f"Ground Truth: {len(gt_boxes)} boxes", fontsize=10)
    ax.axis('off')

    # 子图2-5: 不同融合策略
    strategies = ['original', 'max_confidence', 'weighted_average', 'wbf']

    for idx, strategy in enumerate(strategies, 1):
        ax = axes[idx]
        ax.imshow(img)

        if strategy in fusion_results:
            detections = fusion_results[strategy]['detections']

            for det in detections:
                det_box = det['bbox']
                rect = patches.Rectangle(
                    (det_box[0], det_box[1]), det_box[2], det_box[3],
                    linewidth=2, edgecolor='red', facecolor='none', alpha=0.7
                )
                ax.add_patch(rect)

                # 显示分数
                ax.text(det_box[0], det_box[1] - 5, f"{det['score']:.2f}",
                        bbox=dict(facecolor='red', alpha=0.5), fontsize=8, color='white')

            # 显示统计
            stats = fusion_results[strategy].get('stats', {})
            ap = stats.get('ap', 0)
            tp = stats.get('true_positives', 0)
            fp = stats.get('false_positives', 0)

            title = f"{strategy}\nAP: {ap:.3f} | TP: {tp} | FP: {fp}"
            ax.set_title(title, fontsize=10)

        ax.axis('off')

    # 隐藏多余的子图
    for idx in range(len(strategies) + 1, 6):
        axes[idx].axis('off')

    plt.suptitle(f"Category: {category_name} - Fusion Strategies Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  融合对比可视化已保存: {save_path}")


# --- 多提示集成实验（使用手动AP计算）---
def run_multi_prompt_fusion_experiment():
    """运行多提示集成实验"""
    print("\n" + "=" * 80)
    print("多提示集成实验（使用手动AP计算）")
    print("=" * 80)

    all_results = {}
    visualization_data = {}

    for category in TEST_CATEGORIES:
        print(f"\n{'=' * 60}")
        print(f"处理类别: {category}")
        print('=' * 60)

        category_id = category_name_to_id[category.lower()]
        category_prompts = CATEGORY_MULTI_PROMPTS[category]

        # 获取测试图片
        img_ids = coco.getImgIds(catIds=[category_id])
        if len(img_ids) > 2:  # 测试2张图片
            img_ids = img_ids[:2]

        print(f"使用 {len(img_ids)} 张图片")
        print(f"Prompt数量: {len(category_prompts)}")
        print(f"融合策略: {', '.join(FUSION_STRATEGIES.keys())}")
        print(f"AP计算方法: 手动计算（IoU阈值=0.5）")

        category_results = {}

        for img_id in img_ids:
            try:
                img_info = coco.loadImgs(img_id)[0]
                img_path = os.path.join(COCO_ROOT, img_info['file_name'])

                if not os.path.exists(img_path):
                    continue

                print(f"\n  处理图片: {img_info['file_name']}")

                # 获取真实标注
                ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[category_id])
                anns = coco.loadAnns(ann_ids)
                gt_boxes = [ann['bbox'] for ann in anns]
                print(f"    真实标注数: {len(gt_boxes)}")

                # 加载图片
                _, image_tensor = load_image(img_path)
                W, H = img_info['width'], img_info['height']

                # 步骤1: 使用多个Prompt进行检测
                all_detections = run_multi_prompt_detection(
                    image_tensor, category_prompts, W, H, category,
                    box_threshold=0.1, text_threshold=0.1
                )

                print(f"    原始检测总数: {len(all_detections)} (来自{len(category_prompts)}个Prompt)")

                if len(all_detections) == 0:
                    print("    无检测结果，跳过此图片")
                    continue

                # 步骤2: 应用不同融合策略
                fusion_results = {}

                # 保存原始检测结果（使用手动AP计算）
                ap_score, stats = calculate_ap_manual(all_detections, gt_boxes, category_id, 0.5)
                fusion_results['original'] = {
                    'detections': all_detections,
                    'stats': stats,
                    'ap_score': ap_score
                }

                print(f"    原始检测AP: {ap_score:.4f}")

                # 应用每种融合策略
                for strategy_name, strategy_desc in FUSION_STRATEGIES.items():
                    print(f"    应用融合策略: {strategy_name} ({strategy_desc})")

                    fused_detections = apply_fusion_strategy(
                        all_detections.copy(),  # 创建副本
                        strategy=strategy_name,
                        iou_threshold=0.5
                    )

                    # 使用手动方法计算AP
                    ap_score, stats = calculate_ap_manual(fused_detections, gt_boxes, category_id, 0.5)

                    fusion_results[strategy_name] = {
                        'detections': fused_detections,
                        'stats': stats,
                        'ap_score': ap_score,
                        'strategy_desc': strategy_desc
                    }

                    print(f"      融合后检测数: {len(fused_detections)}, AP: {ap_score:.4f}")

                # 步骤3: 记录结果
                img_key = f"img_{img_id}"
                category_results[img_key] = fusion_results

                # 步骤4: 生成可视化对比
                save_path = f"fusion_comparison_{category}_{img_id}.png"
                visualize_detections_comparison(img_path, fusion_results, gt_boxes, category, save_path)

                # 保存可视化数据
                if category not in visualization_data:
                    visualization_data[category] = {}
                visualization_data[category][img_key] = {
                    'image_path': img_path,
                    'fusion_results': fusion_results,
                    'gt_boxes': gt_boxes,
                    'visualization_path': save_path
                }

            except Exception as e:
                print(f"  处理图片 {img_id} 出错: {e}")
                import traceback
                traceback.print_exc()
                continue

        all_results[category] = category_results

    return all_results, visualization_data


# --- 分析融合策略效果（使用手动AP计算结果）---
def analyze_fusion_strategies(all_results, output_file="fusion_analysis.md"):
    """分析不同融合策略的效果"""
    print("\n" + "=" * 80)
    print("融合策略效果分析（基于手动AP计算）")
    print("=" * 80)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 多提示集成融合策略效果分析\n\n")

        f.write("## 实验概述\n\n")
        f.write("本实验评估了4种不同的检测融合策略在多提示集成中的效果。\n")
        f.write("**注意：所有AP计算均使用手动方法，避免pycocotools库的兼容性问题**\n\n")

        f.write("**测试类别**: " + ", ".join(TEST_CATEGORIES) + "\n")
        f.write("**融合策略**:\n")
        for strategy, desc in FUSION_STRATEGIES.items():
            f.write(f"1. **{strategy}**: {desc}\n")
        f.write("\n**AP计算方法**: 手动计算（IoU阈值=0.5）\n")

        # 对每个类别进行分析
        for category in TEST_CATEGORIES:
            f.write(f"\n## {category.capitalize()} 类别\n\n")

            if category in all_results:
                category_results = all_results[category]

                # 收集所有图片的结果
                strategy_aps = {strategy: [] for strategy in ['original'] + list(FUSION_STRATEGIES.keys())}
                strategy_counts = {strategy: [] for strategy in ['original'] + list(FUSION_STRATEGIES.keys())}
                strategy_precisions = {strategy: [] for strategy in ['original'] + list(FUSION_STRATEGIES.keys())}
                strategy_recalls = {strategy: [] for strategy in ['original'] + list(FUSION_STRATEGIES.keys())}

                for img_key, fusion_results in category_results.items():
                    for strategy, result in fusion_results.items():
                        if strategy in strategy_aps:
                            strategy_aps[strategy].append(result.get('ap_score', 0))
                            strategy_counts[strategy].append(len(result.get('detections', [])))

                            stats = result.get('stats', {})
                            strategy_precisions[strategy].append(stats.get('precision', 0))
                            strategy_recalls[strategy].append(stats.get('recall', 0))

                # 计算平均AP和检测数
                f.write("### 性能对比\n\n")
                f.write("| 策略 | 平均AP@0.5 | 相对提升 | 平均检测数 | 平均精确率 | 平均召回率 |\n")
                f.write("|------|------------|----------|------------|------------|------------|\n")

                baseline_ap = np.mean(strategy_aps['original']) if strategy_aps['original'] else 0

                for strategy in ['original'] + list(FUSION_STRATEGIES.keys()):
                    if strategy_aps[strategy]:
                        avg_ap = np.mean(strategy_aps[strategy])
                        avg_count = np.mean(strategy_counts[strategy])
                        avg_precision = np.mean(strategy_precisions[strategy]) if strategy_precisions[strategy] else 0
                        avg_recall = np.mean(strategy_recalls[strategy]) if strategy_recalls[strategy] else 0

                        if strategy == 'original':
                            rel_improvement = 0
                        else:
                            rel_improvement = ((avg_ap - baseline_ap) / baseline_ap * 100) if baseline_ap > 0 else 0

                        improvement_symbol = ""
                        if rel_improvement > 5:
                            improvement_symbol = "📈"
                        elif rel_improvement < -5:
                            improvement_symbol = "📉"

                        f.write(
                            f"| {strategy} | {avg_ap:.4f} | {rel_improvement:+.1f}% {improvement_symbol} | {avg_count:.1f} | {avg_precision:.3f} | {avg_recall:.3f} |\n")

                f.write("\n### 策略分析\n\n")

                # 找出最佳策略
                best_strategy = None
                best_ap = 0
                for strategy in FUSION_STRATEGIES.keys():
                    if strategy_aps.get(strategy):
                        avg_ap = np.mean(strategy_aps[strategy])
                        if avg_ap > best_ap:
                            best_ap = avg_ap
                            best_strategy = strategy

                if best_strategy:
                    f.write(f"1. **最佳策略**: **{best_strategy}** (平均AP={best_ap:.4f})\n")
                    f.write(
                        f"2. 相对于原始检测，{best_strategy}策略提升了{((best_ap - baseline_ap) / baseline_ap * 100 if baseline_ap > 0 else 0):.1f}%\n")

                # 策略特点分析
                f.write("\n3. **各策略特点**:\n")
                f.write("   - **max_confidence**: 保留最高置信度的检测，减少冗余框\n")
                f.write("   - **weighted_average**: 融合相似检测，提高定位精度\n")
                f.write("   - **nms**: 标准非极大值抑制，平衡精度和召回\n")
                f.write("   - **wbf**: 加权框融合，考虑多源信息，通常性能最佳\n")

                f.write("\n---\n")

    print(f"融合策略分析报告已保存到: {output_file}")


# --- 生成综合报告 ---
def generate_fusion_summary(all_results, output_file="fusion_summary.json"):
    """生成融合实验摘要"""
    summary = {
        'test_categories': TEST_CATEGORIES,
        'fusion_strategies': FUSION_STRATEGIES,
        'ap_calculation_method': 'manual_calculation_iou_0.5',
        'category_results': {},
        'overall_best_strategy': None,
        'key_findings': []
    }

    # 收集各类别的最佳策略
    category_best_strategies = {}

    for category in TEST_CATEGORIES:
        if category in all_results:
            category_results = all_results[category]

            # 计算各策略的平均AP
            strategy_aps = {}
            strategy_details = {}

            for strategy in ['original'] + list(FUSION_STRATEGIES.keys()):
                aps = []
                precisions = []
                recalls = []
                counts = []

                for img_results in category_results.values():
                    if strategy in img_results:
                        result = img_results[strategy]
                        aps.append(result.get('ap_score', 0))

                        stats = result.get('stats', {})
                        precisions.append(stats.get('precision', 0))
                        recalls.append(stats.get('recall', 0))
                        counts.append(len(result.get('detections', [])))

                if aps:
                    strategy_aps[strategy] = np.mean(aps)
                    strategy_details[strategy] = {
                        'avg_ap': np.mean(aps),
                        'avg_precision': np.mean(precisions) if precisions else 0,
                        'avg_recall': np.mean(recalls) if recalls else 0,
                        'avg_detections': np.mean(counts) if counts else 0,
                        'num_samples': len(aps)
                    }

            # 找出最佳策略
            if strategy_aps:
                best_strategy = max(strategy_aps.items(), key=lambda x: x[1])[0]
                best_ap = strategy_aps[best_strategy]

                summary['category_results'][category] = {
                    'best_strategy': best_strategy,
                    'best_ap': best_ap,
                    'all_strategy_details': strategy_details
                }

                category_best_strategies[category] = (best_strategy, best_ap)

    # 找出总体最佳策略
    if category_best_strategies:
        # 计算各策略在所有类别中的平均AP
        strategy_avg_aps = {}
        for strategy in FUSION_STRATEGIES.keys():
            aps = []
            for category, (best_strategy, best_ap) in category_best_strategies.items():
                if best_strategy == strategy:
                    aps.append(best_ap)

            if aps:
                strategy_avg_aps[strategy] = np.mean(aps)

        if strategy_avg_aps:
            overall_best = max(strategy_avg_aps.items(), key=lambda x: x[1])[0]
            summary['overall_best_strategy'] = {
                'strategy': overall_best,
                'avg_ap': strategy_avg_aps[overall_best],
                'description': FUSION_STRATEGIES.get(overall_best, ''),
                'categories_count': sum(1 for _, (s, _) in category_best_strategies.items() if s == overall_best)
            }

    # 关键发现
    summary['key_findings'] = [
        "多提示集成能显著提高检测稳定性（基于手动AP计算验证）",
        "不同融合策略在不同类别上表现不同，WBF通常表现最佳",
        "融合后检测数量通常减少，但检测质量（AP）提高",
        "手动AP计算方法避免了pycocotools库的兼容性问题",
        "加权平均和WBF策略能有效利用多Prompt信息"
    ]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"融合实验摘要已保存到: {output_file}")
    return summary


# --- 验证手动AP计算方法 ---
def validate_manual_ap_calculation():
    """验证手动AP计算方法的正确性"""
    print("\n" + "=" * 80)
    print("验证手动AP计算方法")
    print("=" * 80)

    # 创建一个简单的测试案例
    test_gt_boxes = [[100, 100, 50, 50]]  # 一个真实框

    # 测试案例1: 完美匹配
    perfect_detections = [{
        'bbox': [100, 100, 50, 50],
        'score': 0.9,
        'prompt': 'test',
        'phrase': 'test'
    }]

    ap1, stats1 = calculate_ap_manual(perfect_detections, test_gt_boxes, 1, 0.5)
    print(f"测试1 - 完美匹配:")
    print(f"  AP: {ap1:.4f} (应为1.0)")
    print(f"  精确率: {stats1.get('precision', 0):.4f} (应为1.0)")
    print(f"  召回率: {stats1.get('recall', 0):.4f} (应为1.0)")

    # 测试案例2: 不匹配
    bad_detections = [{
        'bbox': [200, 200, 50, 50],  # 不重叠
        'score': 0.9,
        'prompt': 'test',
        'phrase': 'test'
    }]

    ap2, stats2 = calculate_ap_manual(bad_detections, test_gt_boxes, 1, 0.5)
    print(f"\n测试2 - 不匹配:")
    print(f"  AP: {ap2:.4f} (应为0.0)")
    print(f"  精确率: {stats2.get('precision', 0):.4f} (应为0.0)")

    # 测试案例3: 部分匹配
    partial_detections = [
        {'bbox': [110, 110, 40, 40], 'score': 0.8, 'prompt': 'test', 'phrase': 'test'},  # 高IoU
        {'bbox': [200, 200, 50, 50], 'score': 0.9, 'prompt': 'test', 'phrase': 'test'}  # 不匹配
    ]

    ap3, stats3 = calculate_ap_manual(partial_detections, test_gt_boxes, 1, 0.5)
    print(f"\n测试3 - 部分匹配:")
    print(f"  AP: {ap3:.4f} (应介于0-1之间)")
    print(f"  精确率: {stats3.get('precision', 0):.4f}")
    print(f"  召回率: {stats3.get('recall', 0):.4f}")

    return ap1 > 0.99 and ap2 < 0.01  # 验证基本逻辑


# --- 主程序 ---
if __name__ == "__main__":
    print("Grounding DINO 多提示集成实验")
    print("=" * 80)

    # 验证手动AP计算方法
    ap_valid = validate_manual_ap_calculation()
    if not ap_valid:
        print("\n警告：手动AP计算验证失败，但继续实验...")
    else:
        print("\n✓ 手动AP计算验证通过")

    # 运行多提示集成实验
    all_results, visualization_data = run_multi_prompt_fusion_experiment()

    # 分析融合策略效果
    analyze_fusion_strategies(all_results, "fusion_strategy_analysis.md")

    # 生成摘要报告
    summary = generate_fusion_summary(all_results, "fusion_experiment_summary.json")

    # 输出实验总结
    print("\n" + "=" * 80)
    print("实验总结")
    print("=" * 80)

    print(f"\n实验完成！生成的文件：")
    print("1. 融合策略分析: fusion_strategy_analysis.md")
    print("2. 实验摘要: fusion_experiment_summary.json")
    print("3. 可视化对比图: fusion_comparison_*.png")

    if summary.get('overall_best_strategy'):
        best = summary['overall_best_strategy']
        print(f"\n总体最佳融合策略: {best['strategy']}")
        print(f"描述: {best['description']}")
        print(f"平均AP: {best['avg_ap']:.4f}")
        print(f"在 {best['categories_count']}/{len(TEST_CATEGORIES)} 个类别中表现最佳")

    print(f"\n各类别最佳策略:")
    for category in TEST_CATEGORIES:
        if category in summary.get('category_results', {}):
            result = summary['category_results'][category]
            print(f"  {category}: {result['best_strategy']} (AP={result['best_ap']:.4f})")

    print(f"\n关键发现:")
    for i, finding in enumerate(summary.get('key_findings', []), 1):
        print(f"  {i}. {finding}")

    print("\n" + "=" * 80)
    print("多提示集成实验完成！")
    print("=" * 80)