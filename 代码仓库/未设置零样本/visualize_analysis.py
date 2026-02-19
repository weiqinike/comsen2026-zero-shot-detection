# visualize_analysis.py
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os


def visualize_detections(image_path, detections, save_path=None):
    """可视化检测结果"""
    # 加载图片
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 原图
    axes[0].imshow(img_array)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # 检测结果
    axes[1].imshow(img_array)
    ax = axes[1]

    colors = plt.cm.rainbow(np.linspace(0, 1, 10))

    for i, det in enumerate(detections):
        bbox = det['bbox']  # [x, y, width, height]
        x, y, w, h = bbox
        score = det['score']
        label = det.get('label', f"cat{det['category_id']}")

        # 绘制边界框
        rect = plt.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor=colors[i % len(colors)],
            facecolor='none'
        )
        ax.add_patch(rect)

        # 添加标签
        label_text = f"{label}: {score:.2f}"
        ax.text(
            x, y - 5, label_text,
            color='white', fontsize=8,
            bbox=dict(facecolor=colors[i % len(colors)], alpha=0.8, edgecolor='none', pad=1)
        )

    axes[1].set_title(f"Detections (Total: {len(detections)})")
    axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"可视化结果保存到: {save_path}")
    else:
        plt.show()


def analyze_results(json_path, coco_img_dir):
    """分析检测结果"""
    with open(json_path, 'r') as f:
        results = json.load(f)

    print("=" * 60)
    print("检测结果分析报告")
    print("=" * 60)

    # 按图片分组
    image_results = {}
    for result in results:
        img_id = result['image_id']
        if img_id not in image_results:
            image_results[img_id] = []
        image_results[img_id].append(result)

    print(f"总图片数: {len(image_results)}")
    print(f"总检测框数: {len(results)}")
    print(f"平均每张图片检测数: {len(results) / len(image_results):.2f}")

    # 分析置信度分布
    scores = [r['score'] for r in results]
    print(f"\n置信度统计:")
    print(f"  最小值: {min(scores):.4f}")
    print(f"  最大值: {max(scores):.4f}")
    print(f"  平均值: {np.mean(scores):.4f}")
    print(f"  中位数: {np.median(scores):.4f}")

    # 置信度分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Detection Confidence Scores')
    plt.grid(True, alpha=0.3)
    plt.show()

    # 类别分布
    category_counts = {}
    for result in results:
        cat_id = result['category_id']
        category_counts[cat_id] = category_counts.get(cat_id, 0) + 1

    # 获取类别名称
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

    print(f"\n{'=' * 60}")
    print("检测最多的10个类别:")
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

    for cat_id, count in sorted_categories[:10]:
        if cat_id == 0:
            cat_name = "unknown"
        elif 1 <= cat_id <= 80:
            cat_name = coco_categories[cat_id - 1]
        else:
            cat_name = f"invalid_{cat_id}"
        print(f"  {cat_name}({cat_id}): {count}次")

    # 未知类别比例
    unknown_count = category_counts.get(0, 0)
    total_count = len(results)
    print(f"\n未知类别比例: {unknown_count}/{total_count} ({unknown_count / total_count * 100:.1f}%)")

    # 可视化几张图片的检测结果
    print(f"\n{'=' * 60}")
    print("可视化示例图片...")

    # 加载COCO获取图片路径
    from pycocotools.coco import COCO
    coco_anno_path = r"C:\Users\24344\GroundingDINO\weights\coco\annotations\annotations\instances_val2017.json"

    if os.path.exists(coco_anno_path):
        coco = COCO(coco_anno_path)

        # 可视化前3张图片
        img_ids = list(image_results.keys())[:3]
        for img_id in img_ids:
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(coco_img_dir, img_info['file_name'])

            if os.path.exists(img_path):
                detections = image_results[img_id]
                print(f"\n图片 {img_id} 的检测结果 ({len(detections)} 个检测):")

                # 按置信度排序
                sorted_dets = sorted(detections, key=lambda x: x['score'], reverse=True)[:5]
                for det in sorted_dets:
                    label = det.get('label', f"cat{det['category_id']}")
                    print(f"  {label}: {det['score']:.3f}")

                # 可视化
                save_path = f"detection_vis_{img_id}.jpg"
                visualize_detections(img_path, detections[:10], save_path)  # 只显示前10个


if __name__ == "__main__":
    # 分析之前的检测结果
    json_path = (r"C:\Users\24344\GroundingDINO\detection_hierarchical\hierarchical_detections.json")
    coco_img_dir = r"C:\Users\24344\GroundingDINO\weights\coco\val2017\val2017"

    if os.path.exists(json_path):
        analyze_results(json_path, coco_img_dir)
    else:
        print(f"检测结果文件不存在: {json_path}")
        print("请先运行检测程序")
