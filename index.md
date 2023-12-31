---
title: 主页
layout: home
nav_order: 0
---

欢迎来到多视觉融合的三维重建项目的文档，此项目是学校综合设计项目的一部分。

## 项目目标

开发一个基于多视觉融合的三维重建平台，培养学生对多视角观察和三维重建领域的深入理解。通过硬件平台的需求分析以及相关选型，提供学生实践机会，通过完成任务，掌握三维重建的关键技术和方法。

利用三维打印机设计实现平台，并自己创建校正工具，完成对系统的校准工作。利用平台捕获数据，并编写相关的数据处理程序，根据问题，提出相关的优化思维，结合相关点云处理 PCL，完成最终的曲面拟合以及相关模型输出以及三维交互展示系统。对于整体软件系统的设计考虑，培养学生的团队合作、问题解决和创新能力。开发一个稳定、可靠的底层容器，用于承载多智能体调度优化引擎的运行环境。

## 项目内容

本平台涉及技术包括：
- 图像去噪与图像增强：使用 OpenCV 库中的图像滤波函数进行噪声去除，如高斯滤波、中值滤波等，应用直方图均衡化、对比度增强等方法，提升图像质量
- 特征提取与匹配模块：
    - 特征提取算法，如 SIFT、SURF、ORB 等
    - 基于特征描述子的最近邻相机校准：使用 OpenCV 中的相机标定方法，获得相机内外参数
    - 三维坐标计算：利用相机参数和几何约束，将图像坐标转换为三维世界坐标
- 稠密点云重建：采用 Poisson 重建算法或结构光扫描等方法，将三维点云数据重建为稠密点云
- 稀疏点云重建：使用三角剖分算法或体素格网重建等方法，生成稀疏点云
- 点云滤波与配准模块：
    - 点云滤波：应用 PCL 库中的滤波算法，如统计滤波、高斯滤波、移动最小二乘等，去除噪声和异常点
    - 点云配准：使用 ICP 算法或特征匹配算法，实现点云之间的对齐和融合
- 交互式可视化展示：利用 Python 可视化库，如 Matplotlib、MayaVi、Three.js 等，展示三维模型和点云数据
- 最后结果分析：评估重建结果的质量和精度，讨论可能的误差来源和改进方法

课程分为三个阶段分别在二年级上、二年级下、三年级上，连续积累完成，最终形成一个完整的硬件平台以及软件工具系统，实现对三维模型的三维采集，以及点云数据分析，基于 AI 的模型处理。

三个阶段的不同任务内容为：
1. 系统需求分析，硬件规划以及软件设计考虑，并通过所提供的相关图片，完成三维重建代码测试，明白其中的参数以及捕获数据所需要的完整数据模型，为下一阶段的硬件平台搭建和实现提供数据需求。
- 学习多视角观察的基本原理和三维重建的基本流程
- 研究并选择适合的三维重建算法，如结构光、立体视觉等
- 实现基于 Python 的三维重建代码，包括图像处理、特征提取与匹配、三维坐标转换、点云重建等
2. 搭建多视角拍摄的硬件平台，基于 ARM 架构，完成多视觉捕获接口的综合设计，包括数据捕获，同步控制，以及相关的部署。最后通过三维打印完成设计系统的框架实现，并通过设计完成校正工具和校正过程，实现对整个系统的三维校正。结合嵌入式实时系统开发需求，设计完成相关流程，实现多视觉系统的采集与数据同步，相关数据融合以及三维点云生成算法。
- 学习多视角拍摄的原理和硬件设备的选择
- 设计和搭建多视角拍摄的硬件平台，包括相机布局、同步控制等
- 进行实际的多视角拍摄，获取一组相关图片集用于后续的三维重建
3. 点云处理与分析，基于 ICP 的点云空间匹配融合，实现曲面拟合以及相关模型三维重建算法。利用PCL完成相关点云模型的处理，包括语义分割以及光滑处理等。最后结合 AI 实现一些模型的匹配与识别，特表是平面以及特殊物体的识别。
- 学习点云数据的表示和处理方法，如滤波、配准、特征提取等
- 实现基于 Python 的点云滤波和配准算法，改善点云质量
- 进行点云数据的分析和可视化，提取有用的信息并进行相应的应用

## 组员

- [陈乐兮](https://github.com/XLZXLZXLZ)
- [李俊呈](https://github.com/JcMarical)
- [李欣怡](https://github.com/lextury)
- [王晨宇](https://github.com/LEZIMAN)
- [王祯一](https://github.com/TempContainer)
- [余丰年](https://github.com/ImOP12138)