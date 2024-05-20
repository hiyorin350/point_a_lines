import numpy as np
import conversions as con
import matplotlib.pyplot as plt
from matplotlib import patches

plot_num = 201

lab_image = np.zeros((1, plot_num, 3))
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

# 201個の要素にそれぞれ値を設定
for i in range(plot_num):
    lab_image[0, i, 0] = 20
    lab_image[0, i, 1] = 100 - (i * 1)

rgb_image = 255 * con.lab_to_rgb(lab_image)

# NaNを設定
for i in range(plot_num):
    for j in range(3):
        if rgb_image[0, i, j] > 255 or rgb_image[0, i, j] < 0:
            rgb_image[0, i, j] = np.NaN

print(rgb_image)

# 全てのRGB値がNaNでない行をフィルタリング
valid_indices = ~np.isnan(rgb_image).any(axis=2)[0]

# フィルタリングされたRGBイメージ
valid_rgb_image = rgb_image[0, valid_indices]

# Rチャンネルの最小値と最大値を見つける
min_r_value = np.min(valid_rgb_image[:, 0])
max_r_value = np.max(valid_rgb_image[:, 0])

print(f"Rチャンネルの最小値: {min_r_value}")
print(f"Rチャンネルの最大値: {max_r_value}")

# 最小値に対応するRGB値を見つける
min_r_indices = np.where(valid_rgb_image[:, 0] == min_r_value)
print("最小値に対応するRGB値:")
for idx in min_r_indices[0]:
    print(f"Index {idx}: {valid_rgb_image[idx]}")

# 最大値に対応するRGB値を見つける
max_r_indices = np.where(valid_rgb_image[:, 0] == max_r_value)
print("最大値に対応するRGB値:")
for idx in max_r_indices[0]:
    print(f"Index {idx}: {valid_rgb_image[idx]}")

hsl_image = con.rgb_to_hsl(rgb_image)
hsl_cartesian = con.hsl_to_cartesian(hsl_image)

x = hsl_cartesian[0, :, 0]
y = hsl_cartesian[0, :, 1]

hsl = patches.Circle((0, 0), 100, facecolor="white", edgecolor="red", label="hsl_color_space")
ax1.add_patch(hsl)

ax1.scatter(x, y)
ax1.plot(x, y, linestyle="solid", color="blue")

# アスペクト比を固定
ax1.set_aspect('equal', adjustable='box')

plt.grid(color='b', linestyle=':', linewidth=0.3)
plt.xlabel('0°vector')
plt.ylabel('90°vector')

plt.show()
