---
layout: default
title: 特征提取与匹配
nav_order: 11
---

# 特征提取与匹配

{: .warning }
这篇文章待完善。

## 算法原理

尺度不变特征变换（Scale-Invariant Feature Transform，SIFT）是一种用于图像处理和计算机视觉中的特征提取和匹配的算法。它的原理可以分为以下几个关键步骤：

####  1.构建尺度空间

原始图像与高斯函数进行卷积运算，构建高斯金字塔

这个过程中，将原图进行高斯平滑，得到降采样后的图片，重复此过程。

在此过程中生成的所有图片构成的图片集合作为高斯金字塔。

然后，将高斯金字塔逐层相减构成高斯差分金字塔

(在代码中，金字塔的具体层数取决于SIFT构造函数中的参数nOctaveLayers)

<img src="Picture\report1\image-20230910123311748.png" alt="image-20230910123311748" style="zoom:25%;" /> 

#### 2.关键点定位

对每个点在高斯差分金字塔中与本层及相邻层的相邻点进行对比，若该点在其中绝对值最大，意味着该点与附近相减时差异最大，则该点可能为潜在的关键点(极值点)

随后通过泰勒展开，从这些潜在的离散关键点中复原出连续空间下的真实极值点

<img src="Picture\report1\image-20230910124111017.png" alt="image-20230910124111017" style="zoom:25%;" /> 

#### 3.噪声过滤与边缘检测

去除掉可能为噪声的点，其特征通常为对比度过低 (即删去差分金字塔中绝对值低于阈值的点，它们可能虽然是极值点，但在图像中的定位效果并不关键)

在图像边缘上的关键点，两个方向上的差异可能很大，它们同样可能不适合作为关键点，因此还需对边缘点进行过滤

这一部分应用二阶海森矩阵对比x和y方向的值差异与阈值的比较作为判断依据
$$
H = \begin{bmatrix}
\frac{\partial^2 f}{\partial x^2}& \frac{\partial^2 f}{\partial xy} \\
\frac{\partial^2 f}{\partial yx}& \frac{\partial^2 f}{\partial y^2} 
\end{bmatrix}
$$
过滤Det(H) < 0 和不满足以下公式的点
$$
\frac{\text{Tr}(H)^2}{\text{Det}(H)} < \frac{(r+1)^2}{r}
$$
过滤

(过滤的对比度阈值由SIFT构造函数中的参数contrastThreshold决定)

(过滤的边缘阈值由SIFT构造函数中的参数edgeThreshold决定)

#### 4.点特征提取

在高斯金字塔上找到该关键点的对应位置，并且以它为圆心，尺度为1.5取一个圆

对每个点划分36个角度范围覆盖360度，每个角度10度

统计该圆内所有像素点的梯度方向、每个点所有方向的梯度幅值并进行高斯滤波（补偿因为没有仿射不变性而产生的特征点不稳定的问题）

对梯度方向直方图进行平滑处理，防止噪声干扰主方向的选取



<img src="Picture\report1\image-20230910133029778.png" alt="image-20230910133029778" style="zoom:33%;" /> 

选取数值最高的一项作为主方向同时保留多个辅方向，若某个方向的幅值达到主方向的80%，把它作为辅方向（当然直方图的一个柱子也是一个角度范围，具体的角度还需要抛物线插值进行拟合找出角度值）

在构建关键点的描述符时，为了使其具有旋转不变性，首先让该关键点的主方向和X轴重合，再用插值法填满所需邻域

<img src="Picture\report1\20230910152130.png" alt="20230910152130" style="zoom:70%;" /> 

然后与计算关键点类似，对邻域内每个像素点计算其梯度幅值和幅角，得到关键点的描述符向量

#### 5.特征匹配

sift算法常用的匹配方法是基于上述特征描述子之间的距离度量，比如欧氏距离

这次实验中我们使用了FLANN匹配器进行特征匹配，其原理为高效近似最近邻匹配。

<img src="Picture\report1\20230910172118.png" alt="20230910172118" style="zoom:40%;" /> 

这里的比值检测法用于筛选匹配点，提高匹配准确率

## 源代码

```cpp
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

//SIFT匹配的算法，包含各语句含义解释
int main() 
{
    // 读取两个图像
    cv::Mat image1 = cv::imread("B21.jpg"); //以灰度图打开pic1
    cv::Mat image2 = cv::imread("B22.jpg");

    // 定义掩码图像
    cv::Mat mask1 = cv::Mat::zeros(image1.size(), CV_8U); //设置掩码大小与图片等大，且全部置零
    cv::Mat mask2 = cv::Mat::zeros(image2.size(), CV_8U);

    // 划定采样区域的矩形
    cv::Rect roi1(100, 30, 300, 350);  //从(100,30)坐标框选(300,500)大小区域作为采样区
    cv::Rect roi2(100, 30, 300, 350);
    mask1(roi1).setTo(255); //将该区域掩码设置为255
    mask2(roi2).setTo(255);

    // 创建SIFT特征检测器
    cv::Ptr<cv::Feature2D> sift = cv::SIFT::create
    (
        1000,
        5,
        0.03,
        10.0,
        1.6,
        false
    );//创建SIFT

    // 检测特征点和计算描述符
    //关键点列表: 图像中的特殊点，具有显著特征的位置，用于定位
    std::vector<cv::KeyPoint> keypoints1, keypoints2; 
    //关键描述列表: 包含了某点及附近向量信息的描述，用来确定唯一点
    cv::Mat descriptors1, descriptors2;               
    //参数(输入图片，掩码，关键点向量(列表)，关键描述向量(列表))
    sift->detectAndCompute(image1, mask1, keypoints1, descriptors1); 
    sift->detectAndCompute(image2, mask2, keypoints2, descriptors2);

    // 使用FLANN匹配器进行特征匹配
    //匹配处理器
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED); 
    //匹配结果列表
    std::vector<cv::DMatch> matches;    
    //匹配结果处理
    matcher->match(descriptors1, descriptors2, matches);

    // 绘制匹配结果
    cv::Mat matchImage;
    cv::drawMatches(image1, keypoints1, image2, keypoints2, matches, matchImage);

    // 显示匹配结果
    cv::imshow("SIFT Matches", matchImage);
    cv::waitKey(0);  
}
```

## 环境配置与实现效果

### 环境配置

> 待补充

### 实现效果

<img src="Picture\report1\image-20230910121502264.png" alt="image-20230910121502264" style="zoom: 80%;" /> 

## 效果分析与参数分析


    SIFT参数：
    nfeatures：要检测的关键点数量，默认为0，设定为1000提高效率
    nOctaveLayers：每个金字塔组中的层数，设定为5提高精度
    contrastThreshold：用于过滤关键点的对比度阈值，默认为0.04，考虑石膏像过渡不太明显，略微降低至0.03。
    edgeThreshold：用于过滤边缘关键点的阈值，默认为10.0。
    sigma：高斯滤波器的初始方差，默认为1.6。

对于不同的参数，以下为不同的实现效果

```
(1500, 5, 0.02, 8.0, 1.6, false)
```

<img src="Picture\Report1\图片3.png" alt="图片3" style="zoom:80%;" /> 

```
(1500, 5, 0.04, 12.0, 1.6, false)
```

<img src="Picture\Report1\图片4.png" alt="图片4" style="zoom: 67%;" /> 
