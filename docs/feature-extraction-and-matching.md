---
layout: default
title: 特征提取与匹配
nav_order: 3
---

# 特征提取与匹配

{: .warning }
这篇文章待完善。

## 算法原理

尺度不变特征变换（Scale-Invariant Feature Transform，SIFT）是一种用于图像处理和计算机视觉中的特征提取和匹配的算法。它的原理可以分为以下几个关键步骤：

1. 尺度空间极值检测：SIFT 首先在多个尺度下对输入图像进行高斯平滑，以创建一个尺度空间金字塔。这是通过应用不同尺度的高斯滤波器来实现的，从而使图像在不同尺度下变得模糊或清晰。然后，SIFT 在每个尺度空间中检测图像中的局部极值点，这些点在尺度和位置上都具有稳定性。这些极值点通常代表了图像中的关键特征。
2. 关键点定位：对于检测到的局部极值点，SIFT 使用高斯差分函数来精确定位关键点的位置。这个过程涉及到计算尺度空间中的极值点在不同尺度下的位置和尺度。
3. 方向分配：对于每个关键点，SIFT 计算一个主要方向，以确保关键点具有旋转不变性。这是通过计算关键点周围像素的梯度方向来实现的。SIFT 通常在关键点周围的邻域内创建一个方向直方图，然后选择具有最大峰值的方向作为关键点的主要方向。
4. 关键点描述：在关键点的主要方向确定后，SIFT 创建一个与关键点相关的局部特征描述符。这个描述符是一个向量，包含了关键点周围区域内像素的信息。
描述符的创建方式是将关键点周围的区域分成小的子区域，然后计算每个子区域内像素的梯度方向和幅值，最终构建一个包含这些信息的向量。
5. 特征匹配：通过比较两个图像的关键点描述符，可以进行特征匹配。常用的匹配方法包括最近邻匹配和最近邻距离比较。
最近邻匹配将一个图像的关键点描述符与另一个图像中的所有关键点描述符进行比较，并选择距离最近的那个作为匹配。
最近邻距离比较则要求最近邻的距离要明显小于次近邻的距离，以确保匹配的准确性。

## 源代码

```cpp
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

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
    /*
    SIFT参数：
    nfeatures：要检测的关键点数量，默认为0，设定为1000提高效率
    nOctaveLayers：每个金字塔组中的层数，设定为5提高精度
    contrastThreshold：用于过滤关键点的对比度阈值，默认为0.04，考虑石膏像过渡不太明显，略微降低至0.03。
    edgeThreshold：用于过滤边缘关键点的阈值，默认为10.0。
    sigma：高斯滤波器的初始方差，默认为1.6。
    */

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

## 实现效果
