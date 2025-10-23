# 图像拼接

## 实验目的
学会使用Opencv进行图像特征检测与匹配，掌握SIFT算法和FLANN匹配器的使用方法，利用单应性矩阵实现图像拼接，完成全景图像的生成。

## 实验内容
### 导入所需的库
```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
```

### 读取图像并转换为灰度图
```python
# 读取图像并转换为灰度图以便于后续处理
img_a = cv2.imread('images/a1.jpg')
img_b = cv2.imread('images/a2.jpg')
gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
```

### 使用SIFT算法检测关键点
```python
# 使用SIFT算法检测关键点并计算描述符
sift = cv2.SIFT_create()
kp_a, des_a = sift.detectAndCompute(gray_a, None)
kp_b, des_b = sift.detectAndCompute(gray_b, None)
```

### 使用FLANN匹配器进行特征点匹配
```python
# 使用FLANN匹配器进行特征点匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=100)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des_a, des_b, k=2)
```

### 应用比率测试筛选优质匹配点
```python
# 应用比率测试筛选出优质匹配点
good_matches = []
for m, n in matches:
    # 当第一个匹配点距离小于第二个匹配点距离的0.7倍时，认为是优质匹配
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
```

### 绘制匹配结果并提取匹配点坐标
```python
# 绘制匹配结果并提取匹配点坐标
matched_keypoints_img = cv2.drawMatches(
    img_a, kp_a,
    img_b, kp_b,
    good_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
# 计算得到匹配点的坐标
src_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
```

### 使用RANSAC算法计算单应性矩阵并进行图像拼接
```python
# 使用RANSAC算法计算单应性矩阵并进行图像拼接
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
h_a, w_a = img_a.shape[:2]
h_b, w_b = img_b.shape[:2]
# 计算拼接后图像的尺寸
pts = np.float32([[0, 0], [0, h_b], [w_b, h_b], [w_b, 0]]).reshape(-1, 1, 2)
dst_corners = cv2.perspectiveTransform(pts, H)
all_corners = np.concatenate((dst_corners, np.float32([[0, 0], [w_a, 0], [w_a, h_a], [0, h_a]]).reshape(-1, 1, 2)), axis=0)
[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
# 使用透视变换拼接图像
fus_img = cv2.warpPerspective(
    img_b,
    translation_matrix @ H,
    (x_max - x_min, y_max - y_min)
)
fus_img[-y_min:h_a - y_min, -x_min:w_a - x_min] = img_a
```

### 可视化结果
```python
# 绘制结果
plt.figure(figsize=(40,40))
plt.subplot(1,4,1)
# 绘制图像时转换颜色空间以正确显示
plt.imshow(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB))
plt.subplot(1,4,2)
plt.imshow(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB))
plt.subplot(1,4,3)
# 显示匹配关键点的图像
plt.imshow(cv2.cvtColor(matched_keypoints_img, cv2.COLOR_BGR2RGB))
plt.subplot(1,4,4)
# 显示拼接后的图像
plt.imshow(cv2.cvtColor(fus_img, cv2.COLOR_BGR2RGB))
```
> ![alt text](image.png)
## 实验结果分析
- SIFT算法能够有效检测图像中的关键点，并生成具有旋转不变性和尺度不变性的特征描述符
- FLANN匹配器相比暴力匹配器具有更快的匹配速度，适合处理大量特征点
- 通过比率测试（Lowe's ratio test）可以有效筛选出高质量的匹配点对，提高匹配的准确性
- RANSAC算法能够从包含噪声的匹配点中鲁棒地估计单应性矩阵，剔除错误匹配
- 透视变换结合单应性矩阵可以将两幅图像无缝拼接成全景图，实现视角的统一

## 实验小结
本次实验通过OpenCV实现了基于特征点匹配的图像拼接算法。首先使用SIFT算法提取两幅图像的特征点和描述符，然后利用FLANN匹配器进行快速特征匹配，通过比率测试筛选优质匹配点。最后使用RANSAC算法计算单应性矩阵，并通过透视变换完成图像拼接。实验过程中深入理解了特征检测、特征匹配、单应性矩阵估计等技术，掌握了全景图像生成的完整流程。
