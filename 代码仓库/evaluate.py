# evaluate_detections.py
import json
import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def load_detections(detection_path):
    """加载检测结果"""
    with open(detection_path, 'r') as f:
        detections = json.load(f)
    return detections


def convert_to_coco_format(detections, image_ids=None):
    """将检测结果转换为COCO评估格式"""
    coco_detections = []

    for det in detections:
        # 跳过未知类别
        if det.get('category_id', 0) == 0:
            continue

        # 如果指定了图片ID列表，只保留指定图片的检测结果
        if image_ids is not None and det['image_id'] not in image_ids:
            continue

        coco_det = {
            "image_id": det['image_id'],
            "category_id": det['category_id'],
            "bbox": det['bbox'],  # [x, y, width, height]
            "score": det['score']
        }
        coco_detections.append(coco_det)

    return coco_detections


def calculate_map(coco_gt, coco_detections, output_dir="evaluation_results"):
    """计算mAP和其他评估指标"""
    os.makedirs(output_dir, exist_ok=True)

    # 加载检测结果
    coco_dt = coco_gt.loadRes(coco_detections)

    # 初始化COCO评估器
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

    # 设置评估参数
    coco_eval.params.maxDets = [1, 10, 100]  # 最大检测数
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 获取详细的评估结果
    stats = coco_eval.stats

    # 创建评估报告
    evaluation_report = {
        "overall_metrics": {
            "AP": float(stats[0]),  # AP @ [IoU=0.50:0.95]
            "AP50": float(stats[1]),  # AP @ IoU=0.50
            "AP75": float(stats[2]),  # AP @ IoU=0.75
            "AP_small": float(stats[3]),  # AP for small objects
            "AP_medium": float(stats[4]),  # AP for medium objects
            "AP_large": float(stats[5]),  # AP for large objects
            "AR_max1": float(stats[6]),  # AR @ maxDets=1
            "AR_max10": float(stats[7]),  # AR @ maxDets=10
            "AR_max100": float(stats[8]),  # AR @ maxDets=100
            "AR_small": float(stats[9]),  # AR for small objects
            "AR_medium": float(stats[10]),  # AR for medium objects
            "AR_large": float(stats[11])  # AR for large objects
        },
        "per_category_metrics": {}
    }

    # 保存评估结果
    report_file = os.path.join(output_dir, "evaluation_report_unknown_person.json")
    with open(report_file, 'w') as f:
        json.dump(evaluation_report, f, indent=2)

    return evaluation_report, coco_eval


def generate_visualizations(coco_eval, output_dir="evaluation_results"):
    """生成评估可视化图表"""
    # 1. 精度-召回曲线
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 平均精度曲线
    axes[0, 0].plot(coco_eval.eval['recall'], coco_eval.eval['precision'][:, :, 0, 2].mean(1), 'b-')
    axes[0, 0].set_xlabel('Recall')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_title('Precision-Recall Curve (IoU=0.50:0.95)')
    axes[0, 0].grid(True, alpha=0.3)

    # 不同IoU阈值下的AP
    iou_thresholds = np.arange(0.5, 0.95, 0.05)
    ap_values = []
    for iou in iou_thresholds:
        coco_eval.params.iouThrs = np.array([iou])
        coco_eval.evaluate()
        coco_eval.accumulate()
        ap_values.append(coco_eval.eval['precision'].mean())

    axes[0, 1].plot(iou_thresholds, ap_values, 'r-o')
    axes[0, 1].set_xlabel('IoU Threshold')
    axes[0, 1].set_ylabel('AP')
    axes[0, 1].set_title('AP at Different IoU Thresholds')
    axes[0, 1].grid(True, alpha=0.3)

    # 检测数量分布
    if hasattr(coco_eval, 'evalImgs'):
        num_detections = [len(eval_img['dtIds']) for eval_img in coco_eval.evalImgs if eval_img is not None]
        axes[1, 0].hist(num_detections, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].set_xlabel('Number of Detections per Image')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Detections per Image')

    # 置信度分布
    confidence_scores = []
    for eval_img in coco_eval.evalImgs:
        if eval_img is not None and 'dtScores' in eval_img:
            confidence_scores.extend(eval_img['dtScores'])

    if confidence_scores:
        axes[1, 1].hist(confidence_scores, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].set_xlabel('Confidence Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Confidence Scores')

    plt.tight_layout()
    vis_file = os.path.join(output_dir, "evaluation_visualizations.png")
    plt.savefig(vis_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"可视化图表保存到: {vis_file}")


def analyze_per_category_performance(coco_gt, coco_dt, output_dir="evaluation_results"):
    """分析每个类别的性能"""
    # 获取所有类别
    cat_ids = coco_gt.getCatIds()
    categories = coco_gt.loadCats(cat_ids)

    category_performance = {}

    for cat in categories:
        # 对该类别进行评估
        coco_eval_cat = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval_cat.params.catIds = [cat['id']]
        coco_eval_cat.evaluate()
        coco_eval_cat.accumulate()

        stats = coco_eval_cat.stats
        category_performance[cat['name']] = {
            "AP": float(stats[0]) if len(stats) > 0 else 0.0,
            "AP50": float(stats[1]) if len(stats) > 1 else 0.0,
            "AP75": float(stats[2]) if len(stats) > 2 else 0.0,
            "AP_small": float(stats[3]) if len(stats) > 3 else 0.0,
            "AP_medium": float(stats[4]) if len(stats) > 4 else 0.0,
            "AP_large": float(stats[5]) if len(stats) > 5 else 0.0,
        }

    # 按AP50排序
    sorted_categories = sorted(category_performance.items(),
                               key=lambda x: x[1]['AP50'],
                               reverse=True)

    # 生成类别性能报告
    category_report = {
        "best_performing": [],
        "worst_performing": [],
        "all_categories": category_performance
    }

    # 最佳表现类别（前10）
    for i, (cat_name, metrics) in enumerate(sorted_categories[:10]):
        category_report["best_performing"].append({
            "rank": i + 1,
            "category": cat_name,
            "AP50": metrics["AP50"],
            "AP": metrics["AP"],
            "detections": metrics["num_detections"]
        })

    # 最差表现类别（后10）
    for i, (cat_name, metrics) in enumerate(sorted_categories[-10:]):
        category_report["worst_performing"].append({
            "rank": len(sorted_categories) - i,
            "category": cat_name,
            "AP50": metrics["AP50"],
            "AP": metrics["AP"],
            "detections": metrics["num_detections"]
        })

    # 保存类别报告
    category_file = os.path.join(output_dir, "category_performance.json")
    with open(category_file, 'w') as f:
        json.dump(category_report, f, indent=2)

    # 生成类别性能条形图
    plt.figure(figsize=(12, 8))

    # 只显示前20个类别的AP50
    top_categories = sorted_categories[:20]
    categories_names = [cat[0] for cat in top_categories]
    ap50_scores = [cat[1]['AP50'] for cat in top_categories]

    bars = plt.barh(range(len(categories_names)), ap50_scores)
    plt.yticks(range(len(categories_names)), categories_names)
    plt.xlabel('AP50 Score')
    plt.title('Top 20 Categories by AP50')

    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars, ap50_scores)):
        plt.text(score + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{score:.3f}', va='center')

    plt.tight_layout()
    category_vis_file = os.path.join(output_dir, "category_performance.png")
    plt.savefig(category_vis_file, dpi=150, bbox_inches='tight')
    plt.close()

    return category_report


def print_evaluation_summary(evaluation_report, category_report):
    """打印评估总结"""
    print("=" * 80)
    print("COCO格式目标检测评估报告")
    print("=" * 80)

    metrics = evaluation_report["overall_metrics"]

    print("\n总体指标:")
    print("-" * 40)
    print(f"平均精度 (mAP @ IoU=0.50:0.95): {metrics['AP']:.4f}")
    print(f"平均精度 (AP50 @ IoU=0.50):     {metrics['AP50']:.4f}")
    print(f"平均精度 (AP75 @ IoU=0.75):     {metrics['AP75']:.4f}")

    print(f"\n不同尺寸目标的精度:")
    print(f"  小目标AP:  {metrics['AP_small']:.4f}")
    print(f"  中目标AP:  {metrics['AP_medium']:.4f}")
    print(f"  大目标AP:  {metrics['AP_large']:.4f}")

    print(f"\n召回率 (AR):")
    print(f"  AR @ maxDets=1:   {metrics['AR_max1']:.4f}")
    print(f"  AR @ maxDets=10:  {metrics['AR_max10']:.4f}")
    print(f"  AR @ maxDets=100: {metrics['AR_max100']:.4f}")

    # 最佳和最差类别
    print("\n" + "=" * 80)
    print("最佳表现类别 (按AP50排序):")
    print("-" * 40)
    for item in category_report.get("best_performing", [])[:5]:
        print(f"{item['rank']:2d}. {item['category']:20s} AP50: {item['AP50']:.3f}  AP: {item['AP']:.3f}")

    print(f"\n最差表现类别:")
    print("-" * 40)
    for item in category_report.get("worst_performing", [])[:5]:
        print(f"{item['rank']:2d}. {item['category']:20s} AP50: {item['AP50']:.3f}  AP: {item['AP']:.3f}")


def main():
    """主评估函数"""
    # 路径设置
    detection_path = r"C:\Users\24344\GroundingDINO\detection_optimized_final\optimized_detections.json"
    coco_anno_path = r"C:\Users\24344\GroundingDINO\weights\coco\annotations\annotations\instances_val2017.json"
    output_dir = r"C:\Users\24344\GroundingDINO\evaluation_results"

    # 检查文件是否存在
    if not os.path.exists(detection_path):
        print(f"错误: 检测结果文件不存在 {detection_path}")
        return

    if not os.path.exists(coco_anno_path):
        print(f"错误: COCO标注文件不存在 {coco_anno_path}")
        return

    print("加载检测结果...")
    detections = load_detections(detection_path)

    print("加载COCO标注...")
    coco_gt = COCO(coco_anno_path)

    print("转换检测结果为COCO格式...")
    # 只评估有检测结果的图片
    image_ids = list(set(d['image_id'] for d in detections))
    coco_detections = convert_to_coco_format(detections, image_ids)

    if not coco_detections:
        print("错误: 没有有效的检测结果")
        return

    print(f"评估 {len(coco_detections)} 个检测框...")

    # 计算mAP
    evaluation_report, coco_eval = calculate_map(coco_gt, coco_detections, output_dir)

    # 加载COCO检测结果用于详细分析
    coco_dt = coco_gt.loadRes(coco_detections)

    # 分析每个类别的性能
    category_report = analyze_per_category_performance(coco_gt, coco_dt, output_dir)

    # 生成可视化图表
    generate_visualizations(coco_eval, output_dir)

    # 打印评估总结
    print_evaluation_summary(evaluation_report, category_report)

    print(f"\n评估完成! 结果保存到目录: {output_dir}")


if __name__ == "__main__":
    main()
