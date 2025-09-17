import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional, List, Tuple
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager

# 尝试加载本地Times New Roman字体文件
try:
    # 添加本地字体文件
    font_path = 'timr45w.ttf'
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()
        print(f"Loaded local font: {font_path}")
    else:
        # 如果本地文件不存在，尝试使用系统的Times New Roman
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
        print("Using system serif font")
except Exception as e:
    # 如果出错，使用默认字体
    print(f"Font loading failed: {e}, using default font")
    plt.rcParams['font.family'] = 'serif'

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

big_font = 18
medium_font = 16
small_font = 12


def extract_attention_modules_with_bias(model):
    """
    从模型中提取所有带有learnable_bias_diagonals的注意力模块

    Args:
        model: 已加载的transformer模型

    Returns:
        包含(layer_name, module)的列表
    """
    attention_modules = []

    for name, module in model.named_modules():
        # 检查是否有learnable_bias_diagonals属性
        if hasattr(module, 'learnable_bias_diagonals'):
            attention_modules.append((name, module))
            print(f"Found layer {len(attention_modules) - 1}: {name}")
            print(f"  - Bias shape: {module.learnable_bias_diagonals.shape}")

    if not attention_modules:
        print("Warning: No attention modules with learnable_bias_diagonals found!")

    return attention_modules


def visualize_bias_diagonals_lines(model, positions_to_show=None, skip_layers=1,
                                   save_path=None, use_log_scale=True, max_plots=None):
    """
    使用折线图可视化bias对角线值

    Args:
        model: 已加载的模型
        positions_to_show: 要显示的相对位置列表，None表示自动选择
        skip_layers: 每隔几层画一个
        save_path: 保存路径（支持.pdf和.png格式）
        use_log_scale: 是否使用log刻度
        max_plots: 最多显示多少个图，None表示不限制
    """
    # 提取所有带bias的注意力模块
    attention_modules = extract_attention_modules_with_bias(model)

    if not attention_modules:
        return

    # 选择要绘制的层
    selected_indices = list(range(0, len(attention_modules), skip_layers))
    if max_plots:
        selected_indices = selected_indices[:max_plots]
    selected_modules = [(attention_modules[i], i) for i in selected_indices if i < len(attention_modules)]

    n_plots = len(selected_modules)
    n_cols = 4
    # 动态计算行数
    n_rows = (n_plots + n_cols - 1) // n_cols  # 向上取整

    # 创建图形，动态调整高度
    fig_height = 3.5 * n_rows  # 每行约3.5英寸高
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, fig_height))

    # 确保axes始终是二维数组，便于索引
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_plots == 1:
        axes = np.array([[axes]])

    axes = axes.flatten()

    # 如果没有指定要显示的位置，自动选择
    if positions_to_show is None:
        positions_to_show = 1024

    # 用于存储y轴范围，以便统一
    all_y_min, all_y_max = float('inf'), float('-inf')

    # 第一遍：收集所有数据范围
    for (layer_name, module), layer_idx in selected_modules:
        with torch.no_grad():
            bias_diagonals = module.learnable_bias_diagonals.detach().float().cpu().numpy()
            if isinstance(positions_to_show, int):
                display_len = min(positions_to_show, bias_diagonals.shape[1])
                bias_to_plot = bias_diagonals[:, :display_len]
            else:
                bias_to_plot = bias_diagonals[:, positions_to_show]

            all_y_min = min(all_y_min, bias_to_plot.min())
            all_y_max = max(all_y_max, bias_to_plot.max())

    # 添加一些边距
    y_margin = (all_y_max - all_y_min) * 0.1
    y_min, y_max = all_y_min - y_margin, all_y_max + y_margin

    # 使用tab20颜色映射
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    # 第二遍：绘制图形
    for plot_idx, ((layer_name, module), layer_idx) in enumerate(selected_modules):
        ax = axes[plot_idx]

        with torch.no_grad():
            # 获取bias对角线参数 [num_heads, max_bias_length]
            bias_diagonals = module.learnable_bias_diagonals.detach().float().cpu().numpy()
            num_heads = bias_diagonals.shape[0]

            # 限制显示的位置数
            if isinstance(positions_to_show, int):
                actual_bias_len = bias_diagonals.shape[1]
                display_len = min(actual_bias_len, bias_diagonals.shape[1])
                x_positions = np.arange(1, display_len + 1)  # 从1开始，避免log(0)
                bias_to_plot = bias_diagonals[:, :display_len]
                # x轴显示范围使用positions_to_show，而不是实际的bias长度
                x_axis_max = positions_to_show
            else:
                x_positions = np.array(positions_to_show) + 1
                bias_to_plot = bias_diagonals[:, positions_to_show]
                x_axis_max = max(x_positions)

            # 为每个头绘制一条线，使用一致的颜色
            for head_idx in range(num_heads):
                ax.plot(x_positions, bias_to_plot[head_idx],
                        label=f'Head {head_idx}',
                        alpha=0.8,
                        linewidth=1.2,
                        color=colors[head_idx % 20])

            # 设置x轴为log刻度
            if use_log_scale:
                ax.set_xscale('log')
                ax.set_xlim(1, x_axis_max)  # 使用x_axis_max而不是display_len

            # 统一y轴范围
            ax.set_ylim(y_min, y_max)
            
            # 设置固定的y轴刻度
            ax.set_yticks([-1.5, -0.75, 0, 0.75, 1.5])

            # 设置标题和标签
            ax.set_title(f'Layer {layer_idx}', fontsize=big_font, fontweight='bold')
            ax.set_xlabel('Relative Position' if use_log_scale else 'Relative Position', fontsize=medium_font)
            ax.set_ylabel('Bias Value', fontsize=medium_font)
            
            # 设置坐标轴刻度标签的字体大小
            ax.tick_params(axis='both', which='major', labelsize=small_font)
            ax.tick_params(axis='both', which='minor', labelsize=small_font)

            # 添加网格
            ax.grid(True, alpha=0.3, linestyle='--', which='both' if use_log_scale else 'major')

            # 添加零线参考
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)

    # 隐藏未使用的子图
    for idx in range(n_plots, n_rows * n_cols):
        axes[idx].set_visible(False)
    
    # 添加全局图例
    # 从第一个子图获取图例句柄和标签
    if n_plots > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        # 在图的底部中央添加图例
        fig.legend(handles, labels, 
                  loc='upper center',
                  bbox_to_anchor=(0.5, -0.01),      # 图的下方
                  ncol=min(len(handles), 8),  # 最多10列
                  fontsize=14,
                  framealpha=0.9,   # 图例背景透明度
                  columnspacing=2,    # 列间距
                  handlelength=2.0)     # 图例线长度

    plt.tight_layout(rect=[0, 0.00, 1, 1])  # 为底部图例留出空间

    # 保存图片
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        with PdfPages(save_path) as pdf:
            pdf.savefig(fig, bbox_inches='tight', dpi=300)
            # 添加元数据
            d = pdf.infodict()
            d['Title'] = 'Bias Diagonal Values Visualization'
            d['Author'] = 'Model Analysis'
            d['Subject'] = 'Learnable Bias Analysis'
            d['Keywords'] = 'Transformer, Attention, Bias'
        print(f"Saved as PDF: {save_path}")

    # plt.show()
    return fig


def quick_bias_analysis(model, max_bias_len=1024, skip_layers=1, save_path="bias_viz/bias_lines.pdf"):
    """
    快速分析模型的bias（优化版本）

    Args:
        model: 已加载的模型
        max_bias_len: 显示的最大偏置长度（默认1024）
        skip_layers: 每隔几层画一个（默认1表示画所有层）
        save_path: 保存路径，支持.pdf和.png格式
    """
    print("=" * 50)
    print("Bias Curve Analysis")
    print("=" * 50)

    # 创建保存目录
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    # 绘制bias曲线
    print("\nCreating bias line plots...")
    fig = visualize_bias_diagonals_lines(
        model,
        positions_to_show=max_bias_len,
        skip_layers=skip_layers,
        save_path=save_path,
        use_log_scale=True
    )

    if save_path:
        print(f"\nVisualization saved to: {save_path}")
    else:
        print("\nVisualization complete!")

    return fig


# 使用示例
# 保存为PDF（推荐）
quick_bias_analysis(model, max_bias_len=1024, skip_layers=2, save_path="bias_viz/bias_curves.pdf")

# 只显示特定数量的层（例如只显示前8层）
# fig = visualize_bias_diagonals_lines(model, positions_to_show=1024, skip_layers=1,
#                                      save_path="bias_viz/bias_8layers.pdf",
#                                      use_log_scale=True, max_plots=8)
