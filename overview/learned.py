import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import os


# 复用：按索引或名字片段选中带 learnable bias 的注意力层
def resolve_bias_module(model, layer):
    cands = [(n, m) for n, m in model.named_modules()
             if hasattr(m, "get_learnable_bias") and hasattr(m, "learnable_bias")]
    if not cands:
        raise RuntimeError("未找到带 learnable bias 的注意力模块。")
    if isinstance(layer, int):
        if not (0 <= layer < len(cands)):
            raise IndexError(f"layer 越界：{layer}，可选 0..{len(cands) - 1}")
        return cands[layer]
    hits = [(n, m) for n, m in cands if layer in n]
    if not hits:
        raise ValueError(f"未找到名字包含 '{layer}' 的模块。")
    return hits[0]


def plot_all_layers_all_heads_loglog(model, seq_len=512, max_lag=None,
                                     bias_log_type='abs', min_lag=1,
                                     layer_subset=None, head_subset=None,
                                     head_aggregation=None, alpha=0.6,
                                     save_pdf=None, figsize=(14, 8), dpi=300,
                                     colormap='tab10', show_layer_legend=False):
    """
    一次性可视化所有层所有头的bias曲线，横纵坐标都取对数

    参数:
    - seq_len: 序列长度
    - max_lag: 最大lag，None则用全部
    - bias_log_type: bias的对数变换类型 ('abs', 'symlog', 'signed')
    - min_lag: 最小lag（避免log(0)），从1开始
    - layer_subset: 层子集，None则用全部，可以是list或slice
    - head_subset: 头子集，None则用全部，可以是list或slice
    - head_aggregation: 头聚合方式 (None, 'mean', 'median', 'max', 'min')
    - alpha: 线条透明度
    - save_pdf: 保存路径
    - figsize: 图片尺寸
    - colormap: 颜色映射
    - show_layer_legend: 是否显示层图例
    """
    # 找到所有bias模块
    all_modules = [(n, m) for n, m in model.named_modules()
                   if hasattr(m, "get_learnable_bias") and hasattr(m, "learnable_bias")]

    if not all_modules:
        raise RuntimeError("未找到带 learnable bias 的注意力模块。")

    print(f"找到 {len(all_modules)} 个bias模块")

    # 确定要分析的层
    if layer_subset is not None:
        if isinstance(layer_subset, slice):
            selected_modules = all_modules[layer_subset]
        elif isinstance(layer_subset, (list, tuple)):
            selected_modules = [all_modules[i] for i in layer_subset]
        else:
            selected_modules = [all_modules[layer_subset]]
    else:
        selected_modules = all_modules

    print(f"分析 {len(selected_modules)} 层")

    # 设置颜色
    colors = plt.cm.get_cmap(colormap)

    plt.figure(figsize=figsize, dpi=dpi)

    def safe_log_transform(values, log_type='abs', eps=1e-12):
        """安全的对数变换，处理零值和负值"""
        values = np.array(values)
        if log_type == 'abs':
            return np.log(np.abs(values) + eps)
        elif log_type == 'symlog':
            return np.sign(values) * np.log(np.abs(values) + 1)
        elif log_type == 'signed':
            return np.sign(values) * np.log(np.abs(values) + eps)
        return values

    layer_names = []
    all_bias_values = []

    for layer_idx, (name, mod) in enumerate(selected_modules):
        # 处理序列长度限制
        current_seq_len = seq_len
        if hasattr(mod, "max_bias_length") and seq_len > mod.max_bias_length:
            current_seq_len = mod.max_bias_length

        with torch.no_grad():
            bias = mod.get_learnable_bias(current_seq_len, current_seq_len).detach().to(torch.float32).cpu()

        H, T, _ = bias.shape
        L = T if max_lag is None else min(max_lag + 1, T)
        L = max(min_lag, L)  # 确保不小于min_lag

        # 构建lag序列 (从min_lag开始，避免log(0))
        lag_range = list(range(min_lag, L))
        if not lag_range:
            continue

        x_log = np.log(lag_range)  # 横坐标取对数

        layer_color = colors(layer_idx / max(1, len(selected_modules) - 1))
        layer_name = name.split('.')[-2] if '.' in name else name  # 简化层名
        layer_names.append(f"L{layer_idx}({layer_name})")

        # 确定要分析的头
        if head_subset is not None:
            if isinstance(head_subset, slice):
                head_indices = list(range(H))[head_subset]
            elif isinstance(head_subset, (list, tuple)):
                head_indices = [h for h in head_subset if 0 <= h < H]
            else:
                head_indices = [head_subset] if 0 <= head_subset < H else []
        else:
            head_indices = list(range(H))

        # 收集该层所有头的数据
        layer_head_data = []

        for head_idx in head_indices:
            mat = bias[head_idx]  # [T, T]

            # 计算每个lag的平均bias
            y_values = []
            for lag in lag_range:
                if lag < T:
                    diag_vals = mat.diagonal(offset=-lag)
                    y_values.append(diag_vals.mean().item())
                else:
                    y_values.append(0)  # 超出范围的lag设为0

            layer_head_data.append(y_values)

        # 头聚合
        if head_aggregation is None:
            # 画所有头
            for head_idx, y_values in enumerate(layer_head_data):
                y_log = safe_log_transform(y_values, bias_log_type)
                all_bias_values.extend(y_log)

                # 线型样式：每层内的头用不同透明度
                head_alpha = alpha * (0.3 + 0.7 * head_idx / max(1, len(layer_head_data) - 1))
                plt.plot(x_log, y_log, color=layer_color, alpha=head_alpha,
                         linewidth=2.5)
        else:
            # 聚合所有头
            layer_head_data = np.array(layer_head_data)  # [num_heads, num_lags]

            if head_aggregation == 'mean':
                y_agg = np.mean(layer_head_data, axis=0)
            elif head_aggregation == 'median':
                y_agg = np.median(layer_head_data, axis=0)
            elif head_aggregation == 'max':
                y_agg = np.max(layer_head_data, axis=0)
            elif head_aggregation == 'min':
                y_agg = np.min(layer_head_data, axis=0)
            else:
                y_agg = np.mean(layer_head_data, axis=0)

            y_log = safe_log_transform(y_agg, bias_log_type)
            all_bias_values.extend(y_log)

            plt.plot(x_log, y_log, color=layer_color, alpha=alpha,
                     linewidth=3)

    # 设置透明背景
    fig = plt.gcf()
    fig.patch.set_alpha(0)
    ax = plt.gca()
    ax.patch.set_alpha(0)
    
    # 隐藏刻度标签但保留刻度位置（这样网格线才能正常显示）
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # 显示网格
    plt.grid(True, alpha=0.3, which='both')
    
    # 隐藏边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # 统计信息
    if all_bias_values:
        print(f"变换后bias值范围: [{np.min(all_bias_values):.4f}, {np.max(all_bias_values):.4f}]")
        print(f"总共绘制了 {len([l for l in plt.gca().get_lines()])} 条曲线")

    plt.tight_layout()

    # 保存
    if save_pdf:
        os.makedirs(os.path.dirname(save_pdf) if os.path.dirname(save_pdf) else '.', exist_ok=True)
        if not save_pdf.endswith('.pdf'):
            save_pdf += '.pdf'
        plt.savefig(save_pdf, format='pdf', dpi=dpi, bbox_inches='tight', transparent=True)
        print(f"Log-Log全景图已保存到: {save_pdf}")

    plt.show()


# 使用示例
if __name__ == "__main__":
    # 示例1：所有层所有头的log-log图
    plot_all_layers_all_heads_loglog(
        model,
        seq_len=512,
        max_lag=256,
        bias_log_type='symlog',  # 推荐用symlog保留正负号
        save_pdf='all_layers_all_heads_loglog.pdf',
        figsize=(14, 8)
    )
