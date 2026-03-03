import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import probabilistic_hough_line
from skimage.feature import canny
from scipy.ndimage import binary_erosion

# 读取频谱图数据（假设是一个numpy数组）
spectrogram = np.random.rand(100, 200)

# 应用边缘检测
edges = canny(spectrogram)

# 二值腐蚀，将边缘变细
edges = binary_erosion(edges)

# 使用概率霍夫变换检测直线
lines = probabilistic_hough_line(edges, threshold=10, line_length=5, line_gap=3)

# 在频谱图上绘制检测到的直线
plt.imshow(spectrogram, cmap='gray')
for line in lines:
    p0, p1 = line
    plt.plot((p0[0], p1[0]), (p0[1], p1[1]), color='red')

plt.title("Spectrogram with Detected Lines")
plt.show()
