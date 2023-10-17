# BA优化

本阶段对先前的代码完成了BA优化，优化了点云的生成



### 算法原理

#### Incremental SFM

增量式（Incremental）SFM是一种基于多视角的三维重建方式

它的运行模式是在三角测量（triangulation）和透视N点（PnP）计算的同时，进行局部捆绑调整（Bundle Adjustment）

相较于传统的全局式SFM，增量式SFM具有更高的鲁棒性，健壮性，当引入不合理的异常图片时，其受到的影响相较于全局式的运动重建，受到的影响更小

但这种每次添加图像后，Incremental SFM都需要进行一次Bundle Adjustment，因而当需要处理的图片量较大时，较大的运算量可能导致其效率较低；同时，由于误差累积的原因，增量式SFM还容易出现"漂移"问题，即在添加图片后，场景中原有的点出现偏移

![1](Picture\Report3\1.png)

#### triangulation（三角化）

三角化原理：三角化的基本原理是通过多个视图中的二维图像点来计算对应的三维点的位置。当我们有两个相机的投影矩阵和它们对应的特征点时，可以利用它们之间的几何约束关系来进行三角化计算。

在三角化过程中，我们将相机的投影矩阵与特征点的像素坐标进行反向投影，将二维图像点重新转换为齐次坐标。然后，通过求解齐次线性方程组，可以推导出三维点的齐次坐标。

1. 首先，假设我们已经通过相机的投影矩阵求解出了旋转矩阵R和平移向量T，表示相机的姿态和位置。

2. 接下来，我们有一个二维特征点x2'在相机2上的坐标，这个坐标经过去畸变处理。

3. 然后，我们使用三角化方法将相机1和相机2的投影矩阵P1和P2与特征点的坐标x1和x2计算得到三维点的齐次坐标X。

4. 现在，我们希望将齐次坐标X转换为非齐次坐标表示的三维点。

   - 首先，我们使用叉积的定义来消除s2，即将x2'与R2X进行叉积运算，得到叉积矩阵

   <img src="Picture\Report3\17.png" alt="7" style="zoom:67%;" />

   - 然后，我们将其转化为齐次方程形式

   <img src="Picture\Report3\18.png" alt="7" style="zoom:67%;" />

5. 接下来，我们使用奇异值分解（SVD）来求解上述方程的解X。通过对矩阵进行SVD分解，我们可以得到方程的最小二乘解，即使得左侧0空间的向量。

6. 最后，我们将得到的X进行归一化，将最后一个元素归一化到1，得到真实世界中的三维点X。

#### Perspective-n-Poin (PnP)

 PnP 算法是一种通过给定 3D点的坐标、对应2D点坐标以及内参矩阵，求解相机的位姿的算法

算法需要从 3D 向 2D 映射的点对作为输入参数，如图示的A-a,B-b点等

<img src="Picture\Report3\7.png" alt="7" style="zoom:67%;" /> 

根据相应的归一化约束条件可以获得关于姿态参数的两个方程

PnP算法要求至少存在6对点对用于估计位姿

在相机已校准的前提下，PnP算法可直接使用线性变换:

1. 对于某个点 P = (X, Y, Z, 1)^T，定义增广矩阵 [R|t] 为一个 3 × 4 的矩阵,包含了旋转与平移信息

<img src="Picture\Report3\2.png" alt="2" style="zoom:50%;" />

   2.化简得到约束条件

<img src="Picture\Report3\3.png" alt="3" style="zoom:50%;" />

   3.为简化,定义 T 的行向量，有

<img src="Picture\Report3\4.png" alt="4" style="zoom:50%;" />

<img src="Picture\Report3\5.png" alt="5" style="zoom:50%;" />

  4.进而得到矩阵

<img src="Picture\Report3\6.png" alt="6" style="zoom:67%;" />

 T 共有 12 维，通过提供的6对匹配点，即可实现矩阵 T 的线性求解

#### Bundle Adjustment

在完成triangulation和PnP后，通过BA算法从视觉重建中提炼出最优的3D模型和相机参数

<img src="Picture\Report3\8.png" alt="8" style="zoom:50%;" />

为计算误差优化量，引入该公式:

 观测值+观测值改正数=近似值+近似值改正数

设点坐标为u，相机投影矩阵为C，三位点坐标为X，于是有以下优化方程

![9](Picture\Report3\9.png)



##### · 优化算法

(1)最速下降法:顺梯度负方向寻找更低的函数值

![10](Picture\Report3\10.png)

最速下降法保证了每次迭代函数都是下降的，但其迭代方向是折线形的，导致了最终的收敛效率很低



(2)**Gauss-Newton算法**

若非线性目标函数f(x)具有二阶连续偏导，在x(k)为其极小点的某一近似，在这一点取f(x)的二阶泰勒展开，即：

![13](Picture\Report3\13.png)

![14](Picture\Report3\14.png)



在NewTon算法基础上取消计算黑塞矩阵，转而用一下式代替

![15](Picture\Report3\15.png)

假设重投影误差eij = |uij - vij| = 0，有

![11](Picture\Report3\11.png)

从此处迭代σ直至其收敛

Gauss-Newton的优势在于收敛效率块，但无法保证每次迭代时都在下降

(3)**LM算法**

LM算法是结二者之长的改进

它将Gauss-Newton的最终求解方程改为

![16](Picture\Report3\16.png)

则当参数λ较大时，算法类似Gauss-Newton，而当λ较小时，算法与最速下降法相近

BA优化通常采取LM算法，通过调整参数λ，保证了每次迭代都是下降的，并且可以快速收敛



### 算法实现

BA算法的核心思想是通过最小化重建误差来优化相机参数和三维点的位置。

该算法会考虑到相机的内部参数（比如焦距和畸变）以及外部参数（相机在空间中的位置和朝向）。它还会考虑到每个观察到的图像中的特征点，并尝试调整三维点的位置，使得在重建结果和观察到的图像之间的投影误差最小化。

它基于了先前实现的SIFT特征匹配生成的点云的基础上，引入ceres库函数进行BA优化



在本项目中，使用的ceres库的解决方案如下:

##### 1.构建代价函数：

首先，定义一个或多个成本函数，其中每个成本函数用于度量实际观测值与模型的预测之间的误差。这些成本函数通常定义为残差，即观测值与预测值之间的差异，通常使用ceres::CostFunction /ceres::AutoDiffCostFunction来构建该函数

##### 2.通过代价函数构建优化问题：

创建一个对象，该对象将包含所有的成本函数以及待优化的参数块。参数块可以包括相机的内外参、三维点的坐标等。将每个成本函数与相关参数块关联起来，以构建整个优化问题的定义。

##### 3.求解问题：

创建一个 ceres::Solver::Options 结构，该结构用于配置优化求解器的参数，例如最大迭代次数、收敛容忍度、线性求解器类型等。这些参数会影响优化的过程和结果。

调用 ceres::Solve 函数，将求解器参数、问题对象传递给该函数。Ceres Solver 将使用选定的优化算法和参数来迭代调整参数块的值，以最小化成本函数的残差。

##### 4.得到结果:

基于以上流程，迭代求解得到最终输出







### 源代码

```c++
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <ceres/ceres.h>

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
    cv::Mat M_r = cv::Mat::zeros(3, 4, CV_64F);//创建一个3x4的矩阵 M_r将旋转
    矩阵
    R.copyTo(M_r(cv::Rect(0, 0, 3, 3)));//将旋转矩阵 R 的数据复制到 M_r 的左上
    角3x3的子矩阵中
    t.copyTo(M_r(cv::Rect(3, 0, 1, 3)));//将平移向量 t 的数据复制到 M_r 的第4
    列中
    cv::Mat M_l = cv::Mat::eye(3, 4, CV_64F);//创建一个3x4的单位矩阵 M_l，用
    来表示左相机的投影矩阵。
    cv::Mat P_l = K * M_l;//相机内参矩阵 K 与左相机的投影矩阵 M_l 相乘得到左相机
    的投影矩阵 P_l
    cv::Mat P_r = K * M_r;//将相机内参矩阵 K 与右相机的投影矩阵 M_r 相乘得到右相
    机的投影矩阵 P_r
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

    // 执行BA优化
    ceres::Problem problem;

    for (size_t i = 0; i < point_3d.rows; i++) {
        const cv::Point3d& point = point_3d.at<cv::Point3d>(i);
        const cv::Point2d& observation = p1[i]; // 或者选择其他观测

        ceres::CostFunction* cost_function = BundleAdjustmentCostFunctor::Create(point, observation);
        problem.AddResidualBlock(cost_function, nullptr, R.ptr(), t.ptr());
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // BA优化后的结果即为summary，将其输出
    std::cout << summary.FullReport() << std::endl;

    return 0;
}
        
        
// 定义BA优化的残差项
struct BundleAdjustmentCostFunctor {
    // 构造函数，传入三维点和对应的二维观测
    BundleAdjustmentCostFunctor(const cv::Point3d& point3d, const cv::Point2d& observation)
        : point3d_(point3d), observation_(observation) {}

    template <typename T>
    // 工具函数，重载()运算符，定义残差项，用于Ceres Solver的优化
    bool operator()(const T* const camera_rotation, const T* const camera_translation, T* residual) const {
        T point3d[3];
        point3d[0] = T(point3d_.x);
        point3d[1] = T(point3d_.y);
        point3d[2] = T(point3d_.z);

        T camera_R[9];
        ceres::AngleAxisToRotationMatrix(camera_rotation, camera_R);

        T camera_T[3];
        camera_T[0] = camera_translation[0];
        camera_T[1] = camera_translation[1];
        camera_T[2] = camera_translation[2];

        T projected_point[2];
        T K[9];
        K[0] = T(K.at<double>(0, 0));
        K[1] = T(K.at<double>(0, 1));
        K[2] = T(K.at<double>(0, 2));
        K[3] = T(K.at<double>(1, 0));
        K[4] = T(K.at<double>(1, 1));
        K[5] = T(K.at<double>(1, 2));
        K[6] = T(K.at<double>(2, 0));
        K[7] = T(K.at<double>(2, 1));
        K[8] = T(K.at<double>(2, 2));

        ceres::MatrixMultiplication(camera_R, point3d, projected_point);
        projected_point[0] = projected_point[0] / projected_point[2];
        projected_point[1] = projected_point[1] / projected_point[2];

        residual[0] = T(observation_.x) - K[0] * projected_point[0] - K[1] * projected_point[1] - K[2];
        residual[1] = T(observation_.y) - K[3] * projected_point[0] - K[4] * projected_point[1] - K[5];

        return true;
    }

    //创建CostFunction
    static ceres::CostFunction* Create(const cv::Point3d& point3d, const cv::Point2d& observation) {
        return new ceres::AutoDiffCostFunction<BundleAdjustmentCostFunctor, 2, 3, 3>(
            new BundleAdjustmentCostFunctor(point3d, observation)
        );
    }

    cv::Point3d point3d_;
    cv::Point2d observation_;
};
```



### 参数解释

BA残差项构造函数，接收点与观测信息

```
BundleAdjustmentCostFunctor(const cv::Point3d& point3d, const cv::Point2d& observation)
	: point3d_(point3d), observation_(observation) {}
```

**point3d__(point3d), observation_(observation)**

1. point3D(point3D): 提供三维点
2. observation_(observation): 提供该点观测信息



针对该点返回对应的ceres工具

```
static ceres::CostFunction* Create(const cv::Point3d& point3d, const cv::Point2d& observation) {
        return new ceres::AutoDiffCostFunction<BundleAdjustmentCostFunctor, 2, 3, 3>(
            new BundleAdjustmentCostFunctor(point3d, observation)
        );
    }
```

**ceres::AutoDiffCostFunction<ResidualType,NumResiduals,可选参数X>**

1. ResidualType: 返回的残差项类型
2. NumResiduals: 残差项的维度，即一个残差向量中包含的元素数量
3. X: 可填写不止一项，对应每个参数块的维度，本代码中使用两个3，分别代表相机维度和点维度



BA优化结果

```
ceres::Solve(options, &problem, &summary);
```

**ceres::Solver::Summary ceres::Solve**

**(const ceres::Solver::Options& options,**  

 **ceres::Problem* problem,**   

 **ceres::Solver::Summary* summary );**

1. const ceres::Solver::Options& options: 一个结构体，用于指定非线性最小化的参数和选项
2. ceres::Problem* problem: Ceres Solver 中的核心数据结构, 包括了要优化的成本函数（CostFunction）以及相应的参数块(ParameterBlock),用来整体判断以最小化成本函数的残差
3. ceres::Solver::Summary* summary: 返回参数，用于返回最小化过程的摘要信息。