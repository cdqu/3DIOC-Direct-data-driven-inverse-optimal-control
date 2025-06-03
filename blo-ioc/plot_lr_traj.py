import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import stats

def detect_outliers(trajectories, method='iqr', alpha=5):
    features = []
    for traj in trajectories:
        x_mean = np.mean(traj[:, 0])
        y_mean = np.mean(traj[:, 1])
        displacement = np.linalg.norm(traj[-1] - traj[0])
        features.append([x_mean, y_mean, displacement])
    features = np.array(features)
    
    # 多维度异常检测
    outliers_mask = np.zeros(len(trajectories), dtype=bool)
    
    for i in range(features.shape[1]):  # 对每个特征单独检测
        if method == 'iqr':
            q1, q3 = np.percentile(features[:, i], [25, 75])
            iqr = q3 - q1
            lower = q1 - alpha*iqr
            upper = q3 + alpha*iqr
            outliers_mask |= (features[:, i] < lower) | (features[:, i] > upper)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(features[:, i]))
            outliers_mask |= z_scores > alpha
    
    clean_idx = np.where(~outliers_mask)[0]
    outlier_idx = np.where(outliers_mask)[0]
    
    return clean_idx, outlier_idx


json_file = 'uo_log_ini.json'

with open(json_file, 'r') as f:
    data = json.load(f)
    
trajectories = np.zeros((len(data), 10, 2))
for i, traj in enumerate(data):
    trajectories[i, :, 0] = traj[0]  # x坐标
    trajectories[i, :, 1] = traj[1]  # y坐标
    
print(f"数据形状: {trajectories.shape} (轨迹数, 点数, 坐标维度)")

clean_idx, outlier_idx = detect_outliers(trajectories, method='iqr', alpha=5)
trajectories = trajectories[clean_idx]
print(len(outlier_idx))

# 2. 计算统计量
mean = np.mean(trajectories, axis=0)
std = np.std(trajectories, axis=0)


# 3. 专用可视化函数
plt.figure(figsize=(8, 6), dpi=120)

# 绘制平均轨迹
plt.plot(mean[:, 0], mean[:, 1], '.-',
        color='blue', linewidth=1.5, label='Learned traj (Mean)', zorder=4)

# plot the expert traj
opt_traj = np.load('./opt_traj_rad_ini.npy', allow_pickle=True).item()
_u_star = np.reshape(opt_traj['u_opt'],(-1, 1), order='F')
uo_plot = _u_star.reshape(2,-1,order='F')
plt.plot(uo_plot[0, :], uo_plot[1, :], '.-', color='grey', linewidth=1.5, label='Expert traj', zorder=4)

# 绘制标准差区域（优化短轨迹显示）
for i in range(mean.shape[0]-1):
    # X方向标准差
    plt.fill_betweenx([mean[i, 1], mean[i+1, 1]],
                     [mean[i, 0]-std[i, 0], mean[i+1, 0]-std[i+1, 0]],
                     [mean[i, 0]+std[i, 0], mean[i+1, 0]+std[i+1, 0]],
                     color='#ff7f0e', alpha=0.15, zorder=2)
        
    # Y方向标准差
    plt.fill_between([mean[i, 0], mean[i+1, 0]],
                    [mean[i, 1]-std[i, 1], mean[i+1, 1]-std[i+1, 1]],
                    [mean[i, 1]+std[i, 1], mean[i+1, 1]+std[i+1, 1]],
                    color='#2ca02c', alpha=0.15, zorder=3)
    

plt.scatter(mean[-1, 0], mean[-1, 1], color='red', 
           s=40, label='End point', zorder=5)
    
# 添加连接线
for i in range(mean.shape[0]-1):
    plt.plot([mean[i, 0], mean[i+1, 0]], 
            [mean[i, 1], mean[i+1, 1]], 
            color='#1f77b4', linestyle='--', alpha=0.5, zorder=4)
    

plt.title('Behavior Imitation', fontsize = 16)
plt.xlabel('X Position', fontsize = 15)
plt.ylabel('Y Position', fontsize = 15)
plt.grid(linestyle=':', alpha=0.7)
plt.legend(loc='upper right', framealpha=1, fontsize = 14)
    
plt.tight_layout()
plt.savefig('traj-curve-1.png',dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
