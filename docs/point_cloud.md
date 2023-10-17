---
layout: default
title: 点云重建
nav_order: 12
---

# 点云重建

## 参数分析

### 1. sift->detectAndCompute(img1, cv::noArray(), kp1, des1);
detectAndCompute是SIFT算法中的一个函数，用于在给定的图像上检测关键点并计算特征描述子。

参数分析：

1. img1：要处理的输入图像。
2. cv::noArray()：表示不使用掩码图像，即不对图像进行额外的掩码处理。
3. kp1：存储检测到的关键点的向量。
4. des1：存储计算得到的特征点的矩阵。

### 2. E = cv::findEssentialMat(p1, p2, K, cv::RANSAC, 0.999, 1.0, mask);
findEssentialMat是OpenCV库中用于计算基础矩阵的函数。基础矩阵是在相机标定和立体视觉中常用的矩阵，用于描述两个相机之间的几何关系。

参数分析：

1. p1和p2是两个点集，分别表示两个图像中的特征点对应的像素坐标。这些点集可以是cv::Mat类型的矩阵或其他适当的数据结构，每个点表示为一个2D坐标。
2. K是相机的内参矩阵，用于将像素坐标转换为相机坐标。它是一个3x3的矩阵，包含了相机的焦距、主点位置和畸变参数等信息。
3. cv::RANSAC是用于估计基础矩阵的方法之一。RANSAC（随机抽样一致性）是一种鲁棒的估计方法，可以去除噪声和异常值的影响。
4. 0.999是RANSAC算法的置信度，表示期望的估计精度。较高的置信度要求更多的迭代次数，以获得更可靠的结果。
5. 1.0是一个阈值参数，用于判断哪些点对应的特征点对被认为是内点（inliers）。具体来说，如果两个点之间的重投影误差小于阈值，则将其视为内点。
6. mask是一个输出参数，用于标记哪些特征点对应的特征点对被认为是内点（值为1），哪些被认为是外点（值为0）。

### 3. cv::recoverPose(E, p1, p2, K, R, t);
recoverPose是OpenCV库中用于从基础矩阵恢复相机姿态的函数。
参数分析：

1. E：基础矩阵，描述了两个相机之间的几何关系。
2. p1和p2：两个图像中的特征点对应的像素坐标，可以是cv::Mat类型的矩阵或适当的数据结构。
3. K：相机的内参矩阵，用于将像素坐标转换为相机坐标。
4. R：输出参数，用于存储恢复的旋转矩阵。
5. t：输出参数，用于存储恢复的平移向量。

## 源代码
### 1.重建点云矩阵的代码
```cpp
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
const int MIN_MATCH_COUNT = 10;

int main() {
    //对每一对图片进行匹配
    std::vector<std::pair<std::string, std::string>> img_pairs = {
        std::make_pair("B21.jpg", "B22.jpg"),
        std::make_pair("B21.jpg", "B23.jpg"),
        std::make_pair("B21.jpg", "B24.jpg"),
        std::make_pair("B21.jpg", "B25.jpg"),
        std::make_pair("B22.jpg", "B23.jpg"),
        std::make_pair("B22.jpg", "B24.jpg"),
        std::make_pair("B22.jpg", "B25.jpg"),
        std::make_pair("B23.jpg", "B24.jpg"),
        std::make_pair("B23.jpg", "B25.jpg"),
        std::make_pair("B24.jpg", "B25.jpg"),
    };
    //镜头内参
    cv::Mat K = (cv::Mat_<double>(3, 3) << 719.5459, 0., 0.,
        0., 719.5459, 0.,
        0., 0., 1.);
    //记录配对组数
    int counter = 0;
 
    //对每一对图片进行处理得到点云
    for (const auto& img_pair : img_pairs) {
        counter++; 
        //读取两个图像文件的数据，并将它们分别存储在img1和img2中，以便后续的图像处理和分析
        cv::Mat img1 = cv::imread(img_pair.first);
        cv::Mat img2 = cv::imread(img_pair.second);

        // SIFT 特征匹配

        cv::Ptr<cv::SIFT> sift = cv::SIFT::create();//创建SIFT

        std::vector<cv::KeyPoint> kp1, kp2;
        cv::Mat des1, des2;
        
        //使用了OpenCV库中的SIFT算法来检测关键点并计算特征点
        sift->detectAndCompute(img1, cv::noArray(), kp1, des1);
        sift->detectAndCompute(img2, cv::noArray(), kp2, des2);

        cv::Ptr<cv::FlannBasedMatcher> flann = cv::FlannBasedMatcher::create();
        std::vector<std::vector<cv::DMatch>> matches; //记录最好的匹配和次好的匹配
        flann->knnMatch(des1, des2, matches, 2);

        //将符合条件的特征点分别存入p1和p2
        std::vector<cv::DMatch> good;
        for (size_t i = 0; i < matches.size(); i++) {
            if (matches[i][0].distance < 0.7 * matches[i][1].distance) {
                good.push_back(matches[i][0]);
            }
        }
        if (good.size() > MIN_MATCH_COUNT) {
            std::vector<cv::Point2f> p1, p2;
            for (size_t i = 0; i < good.size(); i++) {
                p1.push_back(kp1[good[i].queryIdx].pt);
                p2.push_back(kp2[good[i].trainIdx].pt);
            }

            // 通过findEssentialMat得到E矩阵
            cv::Mat E, mask;
            E = cv::findEssentialMat(p1, p2, K, cv::RANSAC, 0.999, 1.0, mask);

            std::vector<uchar> matchesMask(mask);

            // 恢复位移矩阵和旋转矩阵
            cv::Mat R, t;
            cv::recoverPose(E, p1, p2, K, R, t);

            // 三角化
            cv::Mat M_r = cv::Mat::zeros(3, 4, CV_64F);//创建一个3x4的矩阵 M_r将旋转矩阵
            R.copyTo(M_r(cv::Rect(0, 0, 3, 3)));//将旋转矩阵 R 的数据复制到 M_r 的左上角3x3的子矩阵中
            t.copyTo(M_r(cv::Rect(3, 0, 1, 3)));//将平移向量 t 的数据复制到 M_r 的第4列中

            cv::Mat M_l = cv::Mat::eye(3, 4, CV_64F);//创建一个3x4的单位矩阵 M_l，用来表示左相机的投影矩阵。

            cv::Mat P_l = K * M_l;//相机内参矩阵 K 与左相机的投影矩阵 M_l 相乘得到左相机的投影矩阵 P_l
            cv::Mat P_r = K * M_r;//将相机内参矩阵 K 与右相机的投影矩阵 M_r 相乘得到右相机的投影矩阵 P_r

            //调用 undistortPoints 函数对特征点 p1 和 p2 进行去畸变处理，
            // 并将结果存储在 p1_un 和 p2_un 中
            // 畸变处理使用相机内参矩阵 K 来校正特征点的畸变。
            std::vector<cv::Point2f> p1_un, p2_un;
            cv::undistortPoints(p1, p1_un, K, cv::noArray());
            cv::undistortPoints(p2, p2_un, K, cv::noArray());

            //将特征点对应的像素坐标通过左右相机的投影矩阵 P_l 和 P_r 进行三角化
            //并将结果存储在齐次坐标矩阵 point_4d_hom 中。
            cv::Mat point_4d_hom;
            cv::triangulatePoints(P_l, P_r, p1_un, p2_un, point_4d_hom);
            
            //将齐次坐标矩阵 point_4d_hom 转换为非齐次坐标表示的三维点云
            cv::Mat point_3d;
            cv::convertPointsFromHomogeneous(point_4d_hom.t(), point_3d);

            //输出点云
            printf("\nthis is point%d\n", counter);
            std::cout << point_3d << std::endl;
           
            
        }
    }
    cv::waitKey(0);
    return 0;
}
```
### 2.在Unity中重建并显示矩阵的代码
```cs
using System;
using System.IO;
using UnityEngine;

public class ShowPointScript : MonoBehaviour
{
    public int order;//展示的组数
    public StreamReader reader;//用于读取保存了矩阵的文件
    private double[] points ;//记录坐标
    public GameObject pointObject;//点所对应的游戏物体
    private string DataPath = "C:\\Users\\LEZIMAN\\Desktop\\outcomes.txt";//文件的路径


    void Start()//Unity内置函数
    {
        FileInfo fInfo = new FileInfo(DataPath);
        if (fInfo.Exists) { reader = new StreamReader( DataPath); }

        string commentOfFile = reader.ReadToEnd();//读取文件中所有文字
        commentOfFile.Trim('\n');//删除换行符
        //每组数据用'#'隔开，以此拆分commentOfFile
        string[] compareStrings = commentOfFile.Split('#');
        //把需要读取的组数据跟据','拆开
        string[] dataString = compareStrings[order].Split(',');
       
        points = new double[dataString.Length];

        for(int i = 0; i < dataString.Length; i++)
        {
          
            points[i] = Convert.ToDouble(dataString[i]);
        }

        for (int i = 0;i<points.Length;i+=3)
        {
            Instantiate(
            pointObject, 
            new Vector3(
            (float)points[i], 
            (float)points[i + 1], 
            (float)points[i + 2])
            *80, 
            Quaternion.identity);
        }

        reader.Dispose();
        reader.Close();
    }

}

```
## 重建结果
以下仅展示匹配效果较好的图片

### 1.B21和B22进行匹配的结果
<img src="Picture\Report2\B21B22_1.png" alt="图片1" style="zoom:40%;" /> 

<img src="Picture\Report2\B21B22_2.png" alt="图片2" style="zoom:40%;" /> 

### 2.B22和B23进行匹配的结果

<img src="Picture\Report2\B22B23_1.png" alt="图片3" style="zoom:40%;" /> 

<img src="Picture\Report2\B22B23_2.png" alt="图片4" style="zoom:40%;" /> 



