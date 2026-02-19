import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'#强制使用cuda
os.environ['TRANSFORMERS_OFFLINE'] = '1'#强制离线

# 首先导入并修补bertwarper
import sys

sys.path.insert(0, '.')

# 在导入任何groundingdino模块之前，打补丁

import matplotlib.pyplot as plt

def monkey_patch_bertwarper():
    """猴子补丁修复bertwarper"""
    import torch

    def safe_generate_masks(tokenized, special_tokens_list, tokenizer):
        """修复版的generate_masks函数"""
        input_ids = tokenized["input_ids"]
        bs, num_token = input_ids.shape

        if num_token == 0:
            # 返回空的tensor
            return (
                torch.zeros((bs, 0), dtype=torch.bool, device=input_ids.device),
                torch.zeros((bs, 0), dtype=torch.bool, device=input_ids.device),
                torch.zeros((bs, 0, 0), dtype=torch.bool, device=input_ids.device)
            )

        # 计算特殊token mask
        special_tokens_mask = torch.zeros((bs, num_token), dtype=torch.bool, device=input_ids.device)
        for special_token in special_tokens_list:
            if isinstance(special_token, int):
                special_tokens_mask |= input_ids == special_token

        # 计算普通token mask
        non_special_tokens_mask = ~special_tokens_mask

        # 创建transfer map
        idx_to_token_id = torch.arange(num_token, device=input_ids.device)
        token_id_to_idx = idx_to_token_id.unsqueeze(0).repeat(bs, 1)

        cate_to_token_mask_list = []
        for i in range(bs):
            cate_to_token_mask_listi = []
            non_special_indices = idx_to_token_id[non_special_tokens_mask[i]]

            for idx in non_special_indices:
                cate_to_token_mask_listi.append(token_id_to_idx[i] == idx)

            # 安全地stack
            if cate_to_token_mask_listi:
                cate_to_token_mask_list.append(torch.stack(cate_to_token_mask_listi, dim=0))
            else:
                # 添加空的tensor
                cate_to_token_mask_list.append(torch.zeros((0, num_token), dtype=torch.bool, device=input_ids.device))

        # 将列表转换为tensor
        transfer_map = torch.stack(cate_to_token_mask_list, dim=0)

        return special_tokens_mask, non_special_tokens_mask, transfer_map

    # 导入并替换函数
    import groundingdino.models.GroundingDINO.bertwarper as bertwarper
    bertwarper.generate_masks_with_special_tokens_and_transfer_map = safe_generate_masks
    print("✅ bertwarper补丁已应用")


# 应用补丁
monkey_patch_bertwarper()

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

BERT_LOCAL_PATH = r"C:\Users\24344\GroundingDINO\bert-base-uncased"

if not os.path.exists(BERT_LOCAL_PATH):
    print(f"❌ BERT路径不存在: {BERT_LOCAL_PATH}")


# 检查文件是否存在
required_files = ['tokenizer_config.json', 'vocab.txt', 'config.json']
missing_files = []
for f in required_files:
    file_path = os.path.join(BERT_LOCAL_PATH, f)
    if not os.path.exists(file_path):
        missing_files.append(f)

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model_with_bert(model_config_path, model_checkpoint_path):
    """修复BERT加载问题"""
    args = SLConfig.fromfile(model_config_path)
    args.device = "cpu"

    # 构建模型
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)

    # 重要：手动设置BERT参数
    print("🔄 加载BERT tokenizer...")
    try:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(
            BERT_LOCAL_PATH,
            local_files_only=True,
            do_lower_case=True
        )
        model.tokenizer = tokenizer
        print("✅ BERT tokenizer加载成功")

        # 测试tokenizer
        test_text = "dog and cat"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"测试tokenization: '{test_text}' -> {tokens.input_ids.shape}")
    except Exception as e:
        print(f"❌ BERT加载失败: {e}")
        print("尝试使用默认tokenizer...")

    # 设置模型为评估模式
    model.eval()
    model.to("cpu")

    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    model = model.to("cpu")
    image = image.to("cpu")
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)


    logits_filt = logits.cpu().clone()
    boxes_filt = boxes.cpu().clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    # 获取短语
    pred_phrases = []
    if hasattr(model, 'tokenizer') and model.tokenizer is not None:
        tokenizer = model.tokenizer
        tokenized = tokenizer(caption)
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
            pred_phrases.append(pred_phrase + f"({logit.max().item():.3f})")
    else:
        # 如果没有tokenizer，使用简单标签
        for i, (logit, box) in enumerate(zip(logits_filt, boxes_filt)):
            pred_phrases.append(f"obj_{i}({logit.max().item():.3f})")

    return boxes_filt, pred_phrases


# 在原有代码中添加BERTWrapper修复
from groundingdino.models.GroundingDINO.bertwarper import \
    generate_masks_with_special_tokens_and_transfer_map as original_generate_masks


def patched_generate_masks(tokenized, special_tokens_list, tokenizer):
    """修补后的mask生成函数"""
    input_ids = tokenized["input_ids"]
    if not hasattr(tokenizer, 'cls_token_id'):
        tokenizer.cls_token_id = tokenizer.cls_token
        tokenizer.sep_token_id = tokenizer.sep_token

    # 确保special_tokens_list有效
    if not special_tokens_list:
        special_tokens_list = [tokenizer.cls_token_id, tokenizer.sep_token_id]

    return original_generate_masks(tokenized, special_tokens_list, tokenizer)


def get_filtered_grounding_output(model, image, caption, box_threshold, text_threshold,
                                  max_detections=10, iou_threshold=0.5):
    """获取过滤后的检测结果"""
    # 获取原始结果
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, caption, box_threshold, text_threshold, cpu_only=True
    )

    if len(boxes_filt) == 0:
        return boxes_filt, pred_phrases

    # 1. 按置信度排序并限制数量
    if len(boxes_filt) > max_detections:
        # 提取每个框的最大置信度
        confidences = []
        for phrase in pred_phrases:
            try:
                # 从"cat(0.856)"中提取0.856
                conf = float(phrase.split('(')[-1].rstrip(')'))
            except:
                conf = 0.0
            confidences.append(conf)

        # 按置信度排序
        sorted_indices = np.argsort(confidences)[::-1]  # 降序
        boxes_filt = boxes_filt[sorted_indices[:max_detections]]
        pred_phrases = [pred_phrases[i] for i in sorted_indices[:max_detections]]

    # 2. 应用非极大值抑制(NMS)去除重叠框
    if len(boxes_filt) > 1 and iou_threshold < 1.0:
        # 转换xywh到xyxy
        boxes_xyxy = torch.zeros_like(boxes_filt)
        boxes_xyxy[:, 0] = boxes_filt[:, 0] - boxes_filt[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_filt[:, 1] - boxes_filt[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_filt[:, 0] + boxes_filt[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_filt[:, 1] + boxes_filt[:, 3] / 2  # y2

        # 计算置信度
        confidences = []
        for phrase in pred_phrases:
            try:
                conf = float(phrase.split('(')[-1].rstrip(')'))
            except:
                conf = 0.0
            confidences.append(conf)

        # 应用NMS
        keep_indices = torch.ops.torchvision.nms(
            boxes_xyxy,
            torch.tensor(confidences),
            iou_threshold
        )

        boxes_filt = boxes_filt[keep_indices]
        pred_phrases = [pred_phrases[i] for i in keep_indices]

    return boxes_filt, pred_phrases


def plot_clean_boxes_to_image(image_pil, tgt, min_confidence=0.2):
    """绘制清理后的边界框"""
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]

    # 过滤低置信度结果
    filtered_boxes = []
    filtered_labels = []

    for box, label in zip(boxes, labels):
        try:
            # 提取置信度
            if '(' in label and ')' in label:
                confidence = float(label.split('(')[-1].rstrip(')'))
                if confidence >= min_confidence:
                    filtered_boxes.append(box)
                    filtered_labels.append(label)
            else:
                filtered_boxes.append(box)
                filtered_labels.append(label)
        except:
            filtered_boxes.append(box)
            filtered_labels.append(label)

    if len(filtered_boxes) == 0:
        print("⚠️ 所有检测结果置信度都低于阈值")
        return image_pil, Image.new("L", image_pil.size, 0)

    print(f"🎨 绘制 {len(filtered_boxes)} 个高置信度框")

    image_copy = image_pil.copy()
    draw = ImageDraw.Draw(image_copy)

    # 为猫和狗选择不同颜色
    colors = {
        'cat': (255, 0, 0),  # 红色
        'dog': (0, 255, 0),  # 绿色
        'animal': (0, 0, 255),  # 蓝色
    }

    for i, (box, label) in enumerate(zip(filtered_boxes, filtered_labels)):
        # 根据标签选择颜色
        color = None
        for key in colors:
            if key in label.lower():
                color = colors[key]
                break

        if color is None:
            color = (255, 165, 0)  # 橙色作为默认

        # 转换坐标
        x_center, y_center, width, height = box

        if isinstance(x_center, torch.Tensor):
            x_center = x_center.item()
            y_center = y_center.item()
            width = width.item()
            height = height.item()

        x0 = int((x_center - width / 2) * W)
        y0 = int((y_center - height / 2) * H)
        x1 = int((x_center + width / 2) * W)
        y1 = int((y_center + height / 2) * H)

        # 确保坐标在合理范围内
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W - 1, x1), min(H - 1, y1)

        # 只绘制足够大的框（过滤掉噪点）
        if (x1 - x0) * (y1 - y0) < 100:  # 面积小于100像素的忽略
            continue

        # 绘制边界框
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

        # 简化标签显示
        if '(' in label and ')' in label:
            simple_label = label.split('(')[0].strip()
            confidence = label.split('(')[-1].rstrip(')')
            display_label = f"{simple_label} ({confidence})"
        else:
            display_label = label

        # 绘制标签
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        draw.text((x0 + 5, y0 + 5), display_label, fill=color, font=font)

    return image_copy


if __name__ == "__main__":
    # ⚠️ 关键修改：降低阈值！
    config_file = r"C:\Users\24344\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py"
    checkpoint_path = r"C:\Users\24344\GroundingDINO\weights\groundingdino_swint_ogc.pth"
    image_path = r"C:\Users\24344\GroundingDINO\.asset\cat_dog.jpeg"

    # 🔥 修改这里！
    text_prompt = "a dog . a cat . animal"  # 更清晰的提示
    output_dir = "outputs"
    box_threshold = 0.2  # 大幅降低！
    text_threshold = 0.2  # 大幅降低！

    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载图片
    print("📷 加载图片...")
    image_pil, image = load_image(image_path)
    print(f"图片尺寸: {image_pil.size}")

    # 2. 加载模型（使用修复版本）
    print("🤖 加载模型...")
    model = load_model_with_bert(config_file, checkpoint_path)

    # 3. 运行推理前检查
    print(f"\n🔍 推理设置:")
    print(f"  文本提示: '{text_prompt}'")
    print(f"  框阈值: {box_threshold}")
    print(f"  文本阈值: {text_threshold}")

    # 4. 运行检测
    with torch.no_grad():
        outputs = model(image[None], captions=[text_prompt])

    # 查看原始输出
    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]

    print(f"\n📊 原始检测结果:")
    print(f"  总检测数: {logits.shape[0]}")
    print(f"  最大置信度: {logits.max().item():.4f}")
    print(f"  平均置信度: {logits.mean().item():.4f}")
    print(f"  置信度>0.05的数量: {(logits.max(dim=1)[0] > 0.05).sum().item()}")
    print(f"  置信度>0.03的数量: {(logits.max(dim=1)[0] > 0.03).sum().item()}")

    # 5. 应用阈值（使用更低的阈值）
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    print(f"\n🎯 过滤后结果 (阈值={box_threshold}):")
    print(f"  保留检测数: {len(logits_filt)}")

    # 6. 获取预测短语
    pred_phrases = []
    if hasattr(model, 'tokenizer') and model.tokenizer is not None:
        tokenizer = model.tokenizer
        tokenized = tokenizer(text_prompt)
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > text_threshold,
                tokenized,
                tokenizer
            )
            confidence = logit.max().item()
            pred_phrases.append(f"{pred_phrase} ({confidence:.3f})")
    else:
        for i, logit in enumerate(logits_filt):
            pred_phrases.append(f"obj_{i} ({logit.max().item():.3f})")

    # 7. 打印详细结果
    if len(boxes_filt) > 0:
        print(f"\n✅ 检测到 {len(boxes_filt)} 个物体:")
        for i, (box, phrase) in enumerate(zip(boxes_filt, pred_phrases)):
            print(f"  物体{i + 1}: {phrase}")
            print(f"    边界框: [{box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f}]")

        # 8. 绘图
        size = image_pil.size
        pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],
            "labels": pred_phrases,
        }

        image_with_box, _ = plot_boxes_to_image(image_pil.copy(), pred_dict)

        # 保存结果
        output_path = os.path.join(output_dir, "final_result.jpg")
        image_with_box.save(output_path)

        # 显示
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image_pil)
        plt.title("原始图片")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(image_with_box)
        plt.title(f"检测结果 ({len(boxes_filt)}个物体)")
        plt.axis('off')
        plt.show()

        print(f"\n✅ 结果已保存到: {output_path}")
    else:
        print(f"\n⚠️ 没有检测到物体！尝试:")
        print("  1. 降低box_threshold到0.03或0.02")
        print("  2. 使用更详细的文本提示")
        print("  3. 检查BERT模型是否加载正确")

        # 显示原始图片
        plt.imshow(image_pil)
        plt.title("原始图片 (未检测到物体)")
        plt.axis('off')
        plt.show()
