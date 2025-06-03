import json
import numpy as np
import matplotlib.pyplot as plt


with open('upper_history_ini.json', 'r') as f:
    data = json.load(f)  # 假设data是30x1000的列表（30条曲线，每条1000点）

# 2. 转换为NumPy数组并计算统计量
curves = np.array(data)  # 形状 (30, 1000)
mean_curve = np.mean(curves, axis=0)
deviations = np.mean(np.abs(curves - mean_curve), axis=1)
valid_curves = curves[deviations < 2 * np.median(deviations)]

lower = np.percentile(valid_curves, 0, axis=0)
upper = np.percentile(valid_curves, 80, axis=0)
curves = np.clip(valid_curves, lower, upper)

mean = np.mean(curves, axis=0)  # 计算均值（沿30条曲线的方向）
std = np.std(curves, axis=0)     # 计算标准差

# 3. 绘制平均值曲线和标准差阴影
plt.figure(figsize=(10, 7))
plt.plot(mean, color='red', label=r"Risk $\mathcal{S}(\theta_k)$ (Mean)")
plt.fill_between(
    x=np.arange(1000),           # 横轴（迭代次数）
    y1=mean - std,               # 下限：均值-标准差
    y2=mean + std,               # 上限：均值+标准差
    color='red', alpha=0.2,     # 半透明阴影
    label=r'Risk (Std dev)'
)


# 1. 从JSON文件中加载数据
with open('upper_grad_ini.json', 'r') as f:
    data = json.load(f)  # 假设data是30x1000的列表（30条曲线，每条1000点）

# 2. 转换为NumPy数组并计算统计量
curves = np.array(data)  # 形状 (30, 1000)
mean_curve = np.mean(curves, axis=0)
deviations = np.mean(np.abs(curves - mean_curve), axis=1)
valid_curves = curves[deviations < 2 * np.median(deviations)]

lower = np.percentile(valid_curves, 0, axis=0)
upper = np.percentile(valid_curves, 70, axis=0)
curves = np.clip(valid_curves, lower, upper)

mean = np.mean(curves, axis=0)  # 计算均值（沿30条曲线的方向）
std = np.std(curves, axis=0)     # 计算标准差

# 3. 绘制平均值曲线和标准差阴影
plt.plot(mean, color='blue', label=r'Gradient $\Vert \nabla \mathcal{S}(\theta_k) \Vert$ (Mean)')
plt.fill_between(
    x=np.arange(1000),           # 横轴（迭代次数）
    y1=mean - std,               # 下限：均值-标准差
    y2=mean + std,               # 上限：均值+标准差
    color='blue', alpha=0.2,     # 半透明阴影
    label=r'Gradient (Std dev)'
)


plt.xlabel('Iteration(k)', fontsize=15)
plt.ylabel('Loss value (Log-scale)', fontsize=15)
plt.title('BLO-IOC Learning Curve', fontsize=16)
plt.legend(fontsize=14)
plt.grid(alpha=0.3)

# 5. 可选：对数坐标（如果需要）
#plt.ylim(3e-5, 30)
#plt.ylim(1e-5, 10)
#plt.xlim(-20, 1000)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.yscale('log')  # 纵轴对数化（适用于跨数量级损失值）
plt.xscale('log')
#plt.savefig('blo-lr-curve.png',          # 文件名
#    dpi=300,               # 分辨率（默认100，推荐300-600）
 #   bbox_inches='tight',   # 去除白边
#    pad_inches=0.1)      # 透明背景（可选

plt.show()