import numpy as np
import matplotlib.pyplot as plt

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_alpha(0)

# 定义8个head的slope值
slopes = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#48DBFB']

# x轴：相对位置距离
x = np.arange(0, 20, 0.1)

# 绘制不同slope的衰减曲线
for slope, color in zip(slopes, colors):
    # ALiBi使用负数偏置，距离越远，偏置越负
    y = -slope * x
    ax.plot(x, y, linewidth=3.5, color=color, alpha=0.8)

# 设置图形属性
ax.patch.set_alpha(0)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, 20)
ax.set_ylim(-10, 0.5)

# 添加水平线在y=0处
ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)

# 隐藏边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# 调整布局
plt.tight_layout()

# 保存为PDF
plt.savefig('alibi_slope_visualization.pdf', dpi=300, bbox_inches='tight', transparent=True)

# 显示图形
plt.show()

print("图形已保存为 alibi_slope_visualization.pdf")