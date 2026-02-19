# regenerate_charts.py
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
from matplotlib import rcParams

# 1. 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 2. 设置高DPI和清晰度
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['font.size'] = 10
rcParams['axes.titlesize'] = 12
rcParams['axes.labelsize'] = 10


def regenerate_charts():
    """重新生成清晰的图表"""

    # 您的数据
    data = {
        'AP_metrics': {
            'AP': 0.00012376,
            'AP50': 0.00012376,
            'AP75': 0.00012376
        },
        'AR_metrics': {
            'max1': 2.0994e-05,
            'max10': 6.7621e-05,
            'max100': 6.7621e-05
        },
        'size_AP': {
            '小目标': 0.00010754,
            '中目标': 0.00011019,
            '大目标': 0.00012376
        },
        'detection_stats': {
            '总图片数': 32,
            '总检测框数': 202,
            '平均检测数': 6.31,
            'dog检测数': 129,
            'bus检测数': 35,
            'motorcycle检测数': 10
        }
    }

    # 创建清晰的图表
    fig = plt.figure(figsize=(16, 10), facecolor='white')

    # 子图1: AP指标柱状图（添加完整标题）
    ax1 = plt.subplot(2, 2, 1)
    categories = list(data['AP_metrics'].keys())
    values = list(data['AP_metrics'].values())

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax1.bar(categories, values, color=colors, edgecolor='black')

    # 完整标题
    ax1.set_title('COCO检测评估指标对比', fontsize=12, fontweight='bold', pad=15)
    ax1.set_ylabel('平均精度 (AP)', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # 添加数值标签（科学计数法显示）
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + height * 0.1,
                 f'{value:.2e}', ha='center', va='bottom', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # 子图2: AR热力图（完整标签）
    ax2 = plt.subplot(2, 2, 2)

    ar_matrix = np.array(list(data['AR_metrics'].values())).reshape(1, -1)
    im = ax2.imshow(ar_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1e-4)

    # 完整标签
    ax2.set_xticks(np.arange(len(data['AR_metrics'])))
    ax2.set_xticklabels(['maxDets=1', 'maxDets=10', 'maxDets=100'], fontsize=9)
    ax2.set_yticks([0])
    ax2.set_yticklabels(['召回率 (AR)'], fontsize=9)
    ax2.set_title('不同检测数量下的召回率', fontsize=12, fontweight='bold', pad=15)

    # 添加清晰数值
    for i, value in enumerate(data['AR_metrics'].values()):
        ax2.text(i, 0, f'{value:.2e}',
                 ha='center', va='center',
                 fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.colorbar(im, ax=ax2, shrink=0.8, label='AR值')

    # 子图3: 雷达图（完整标签）
    ax3 = plt.subplot(2, 2, 3, projection='polar')

    size_categories = list(data['size_AP'].keys())
    size_values = list(data['size_AP'].values())

    # 归一化
    max_val = max(size_values)
    normalized = [v / max_val for v in size_values] if max_val > 0 else size_values

    # 闭合雷达图
    angles = np.linspace(0, 2 * np.pi, len(size_categories), endpoint=False).tolist()
    normalized += normalized[:1]
    angles += angles[:1]

    ax3.plot(angles, normalized, 'o-', linewidth=2, color='#FF9F43')
    ax3.fill(angles, normalized, alpha=0.25, color='#FF9F43')

    # 完整标签
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(size_categories, fontsize=9)
    ax3.set_title('不同尺寸目标检测精度对比', fontsize=12, fontweight='bold', pad=15)
    ax3.grid(True)

    # 添加数值标签
    for angle, label, value in zip(angles[:-1], size_categories, size_values):
        x = np.cos(angle) * 1.1
        y = np.sin(angle) * 1.1
        ax3.text(x, y, f'{value:.2e}', ha='center', va='center', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # 子图4: 检测统计条形图（完整标签）
    ax4 = plt.subplot(2, 2, 4)

    stats_categories = list(data['detection_stats'].keys())
    stats_values = list(data['detection_stats'].values())

    colors_stats = plt.cm.Set3(np.linspace(0, 1, len(stats_categories)))
    bars = ax4.barh(stats_categories, stats_values, color=colors_stats, edgecolor='black')

    ax4.set_title('检测结果统计', fontsize=12, fontweight='bold', pad=15)
    ax4.set_xlabel('数量', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='x')

    # 添加数值标签
    for bar, value in zip(bars, stats_values):
        width = bar.get_width()
        ax4.text(width + max(stats_values) * 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{value}', ha='left', va='center', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # 设置全局标题
    plt.suptitle('GroundingDINO在COCO数据集上的检测性能评估',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()

    # 保存高质量图片
    plt.savefig('clear_evaluation_charts.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('clear_evaluation_charts.pdf', bbox_inches='tight')

    print("已生成清晰的图表:")
    print("1. clear_evaluation_charts.png (300 DPI)")
    print("2. clear_evaluation_charts.pdf (矢量格式)")
    print("\n主要改进:")
    print("- 完整的中文标签和标题")
    print("- 更高的分辨率和清晰度")
    print("- 添加了数值标签")
    print("- 优化了颜色对比度")


if __name__ == "__main__":
    regenerate_charts()
