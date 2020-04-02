# 一、研究背景、概况及意义

## 选题与研究意义

交通事故是当前世界各国所面临的严重社会问题之一，已被公认为当今世界危害人类生命安全的第一大公害，每年因交通事故的原因至少使50万人死亡. 欧美各国的交通事故统计分析表明，交通事故中80%～90%是人的因素造成的. 根据美国国家公路交通安全署的统计，在美国的公路上，每年由于司机在驾驶过程中跌入睡眠状态而导致大约10万起交通事故，约有1500起直接导致人员死亡，711万起导致人员伤害.在欧洲的情况也大致相同，如在德国境内的高速公路上25%导致人员伤亡的交通事故，都是由疲劳驾驶引起的. 根据2001年中国交通部的统计，我国48 %的车祸由驾驶员疲劳驾驶引起，直接经济损失达数十万美元. 有关汽车驾驶员的疲劳检测问题，随着高速公路的发展和车速的提高，目前已成为汽车安全研究的重要一环。由此可见，疲劳驾驶导致的危害非常严重。所以，深入研究疲劳驾驶以及其监测系统是非常有必要的。

## 设计思路的综述

本课题我们要设计一个通过摄像头中获取人脸图像然后进行处理，采集人眼的特征点的识别。
1.我们依靠眼睛的纵横比例（EAR）来确定一个人是否眨眼。
2.我们将利用Python，OpenCV和dlib来执行面部标志检测和检测视频流中的人眼。
3.基于代码，我们将应用我们的方法来检测示例摄像头以及视频文件流中的人眼识别。此设计可以用于疲劳驾驶与多种创新用途。

# 二、研究主要内容

## 1.人脸特征点

在讨论EAR之前，先来看看68个人脸特征点：

<img src="https://img-blog.csdnimg.cn/20191203220018781.jpg"   width="700px">

我们可以应用脸部特征点检测来定位脸部的区域，包括眼睛，眉毛，鼻子，耳朵和嘴巴，这也意味着我们可以通过需要的脸部区域来提取需要的脸部结构，包括我们所需要的人眼区域。
人脸特征点检测本身的算法是很复杂的，dlib库中给出了相关的实现。

## 2.眼睛纵横比（EAR）

眼睛纵横比（EAR）它是Soukupová和Čech在其2016年的论文“Real-Time Eye Blink Detection using Facial Landmarks”中的提出的一个概念模型。
今天介绍的这个方法与传统的计算眨眼图像处理方法是不同的，使用眼睛的长宽比是更为简洁的解决方案，它基于眼睛的特征点之间的距离比例是一个非常简单的计算。

<img src="https://img-blog.csdnimg.cn/20191203220403965.png"   width="400px">

上图中的6个特征点p1至p6是人脸特征点中对应眼睛的6个特征点。 每只眼睛由6个（x，y）坐标表示，基于这个描述，我们抓住重点：这些坐标的宽度和高度之间有一个关系。
顺理成章地，我们可以导出EAR的方程： 

<img src="https://img-blog.csdnimg.cn/20191203220433252.png"   width="300px">

眼睛纵横比（EAR）关系方程
```python
def eye_aspect_ratio(eye):# 计算EAR
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
```
眼睛纵横比（EAR）编程python代码

这个方程的分子是计算垂直眼睛标志之间的距离，而分母是计算水平眼睛标志之间的距离，由于水平点只有一组，而两组垂直点，所以分母乘上了2，以保证两组特征点的权重相同。使用这个简单的方程，我们可以避免使用图像处理技术，简单地依靠眼睛地标距离的比例来确定一个人是否眨眼。

<img src="https://img-blog.csdnimg.cn/20191203220542353.png"   width="400px">

在上图中绘出了眼纵横比随时间的视频剪辑曲线图。正如我们所看到的，眼睛纵横比是恒定的，然后迅速下降到接近零，然后眼睛睁开后增加。由此我们可以根据此方法来判断眨眼频率和时间，以此来检测驾驶人是否存在疲劳驾驶。

## 3.代码各个模块
(1).导入各模块

```python
import numpy as np 
import cv2
import dlib
from scipy.spatial import distance
import os
from imutils import face_utils
import matplotlib.pyplot as pl
import winsound
```

(2).计算EAR模块

```python
pwd = os.getcwd()# 获取当前路径
model_path = os.path.join(pwd, 'model')# 模型文件夹路径
shape_detector_path = os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat')# 人脸特征点检测模型路径
detector = dlib.get_frontal_face_detector()# 人脸检测器 
predictor = dlib.shape_predictor(shape_detector_path)# 人脸特征点检测器
```

(3).导入检测器模块

```python
pwd = os.getcwd()# 获取当前路径
model_path = os.path.join(pwd, 'model')# 模型文件夹路径
shape_detector_path = os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat')# 人脸特征点检测模型路径
detector = dlib.get_frontal_face_detector()# 人脸检测器 
predictor = dlib.shape_predictor(shape_detector_path)# 人脸特征点检测器
```

(4).定义模块

```python
EYE_EAR = 0.2# EAR阈值
EYE_EAR_BEYOND_TIME = 15# 设置眨眼超时时间，来累加眨眼超时次数
EYE_EAR_WARNIG = 2# 设置眨眼超时次数超过次数,给予提醒警告
WARNIG = 0

RIGHT_EYE_START = 37 - 1# 对应特征点的序号
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1

frame_counter = 0# 连续帧计数
blink_counter = 0# 眨眼计数

duration = 1000  # 眨眼超时持续时间
duration2 = 5000  # 警告持续时间
freq = 440  # 报警声频率

ey = []# 定义一个 y 轴的空列表用来接收动态的数据
```

(5).打开模块

```python
plt.ion()# 开启一个画图窗口

leftEAR_TXT = open("data/leftEAR.txt","w")# 打开保存data的文本
rightEAR_TXT = open("data/rightEAR.txt","w")
EAR_TXT = open("data/ear.txt","w")
```
(6).视频流导入与处理模块

```python
while(1):
	ret, img = cap.read()# 读取视频流的一帧
	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# 转换为灰阶
	rects = detector(gray, 0)# 人脸检测

	for rect in rects:

		shape = predictor(gray, rect)# 检测特征点
		points = face_utils.shape_to_np(shape)# 将面部界标（x，y）坐标转换为NumPy数组
		leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]# 取出特征点
		rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]
		leftEAR = eye_aspect_ratio(leftEye)# 计算左右眼
		rightEAR = eye_aspect_ratio(rightEye)

		ear = (leftEAR + rightEAR) / 2.0 # 求平均ear

		leftEyeHull = cv2.convexHull(leftEye)# 寻找轮廓
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(img, [leftEyeHull], -1, (192,255,62), 1)# 绘制轮廓
		cv2.drawContours(img, [rightEyeHull], -1, (192,255,62), 1)

```

(7).判断语句模块

```python
		if ear < EYE_EAR: # 如果EAR小于阈值，开始累计连续眨眼次数
			frame_counter += 1
		else:
			if frame_counter >= EYE_EAR_BEYOND_TIME:# 连续帧计数超过EYE_EAR_BEYOND_TIME的帧数时，累加超时次数（blink_counter+1）并给予提示警告
				winsound.Beep(freq, duration)# 提示报警声
				blink_counter += 1
				frame_counter = 0
			if blink_counter >= EYE_EAR_WARNIG:# 连续帧计数超过EYE_EAR_WARNIG帧数时，给警告次数（WARNIG+1）并给予长时间报警警告
				winsound.Beep(freq, duration2)# 长时间报警声
				WARNIG += 1
				blink_counter = 0
```

(8).输出显示模块

```python
		# 在图像上显示EAR的值，Exceeding-number超过次数，WARNIG警告次数
		cv2.putText(img, "WARNIG!:{0}".format(WARNIG), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
		cv2.putText(img, "Exceeding-number:{0}".format(blink_counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
		cv2.putText(img, "EAR:{:.2f}".format(ear), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
		print('leftEAR = {0}'.format(leftEAR))# 直接输出值
		print('rightEAR = {0}'.format(rightEAR)) 
		print('ear = {0}'.format(ear))
		print('-'*20)
		cv2.imshow("KZero", img)# 显示视频窗口

		ey.append(ear)# 添加 i 的平方到 y 轴的数据中
		plt.clf()# 清除之前画的图
		plt.plot(ey)# 画出当前值的图形
		plt.pause(0.01)# 暂停一秒
```

(9).数据保存模块

```python
		print('{0}'.format(leftEAR),file=leftEAR_TXT)# 保存左右眼ear和ear的数据
		print('{0}'.format(rightEAR),file=rightEAR_TXT)
		print('{0}'.format(ear),file=EAR_TXT)
```

(10).退出模块

```python
if cv2.waitKey(1) & 0xFF == 27:# 按esc退出
		break
		plt.ioff() # 关闭画图的窗口

cap.release()
cv2.destroyAllWindows()
```

# 三、研究步骤、方法及措施

## 1.硬件设备:笔记本电脑、普通摄像头。

## 2.环境安装配置步骤: （基于Win10）
**环境搭建：**
(1)右击“我的电脑”属性点击“高级系统设
(2)点击“环境变量”
(3)系统变量里知道“Path”点击编辑 
(4)点击“新建”把需要搭建的路径复制进去点击确定
**1.Phon3.7安装与环境搭建** 官方网：https://www.python.org/
**2.Opencv4.1 安装与环境搭建** 官方网：https://opencv.org/
**3.安装pip 19.2.3**（它是Python包管理工具，提供了Python包的查找、下载、安装、卸载的功能，有利于之后的环境搭建。）
**（以下操作都可以用pip进行安装，除了dlib的特殊情况）**
Win+R窗口输入“cmd”，然后输入对应的库安装，
**4.安装opencv-python 4.1.1.26 计算机视觉库。**
在cmd里输入在线安装“pip install opencv-python”
**5.安装Numpy 1.17.2 开源的数值计算扩展，功能是科学计算，数据分析与处理。**
“pip install Numpy”
**6.安装Scipy 1.3.1 距离计算库。**
“pip install Scipy”
**7.安装imutils 0.5.3图像处理工具包。**
“pip install imutils”
**8.安装matplotlib 3.1.1 python的2D绘图库。**
“pip install matplotlib”
**9.安装dlib 19.17.0人脸识别库（由于WIN10的特殊情况，dlib库需要C++的环境，搭建dlib库之前我们需要安装Visual Studio、Cmake、boost进行C++的环境搭建。）**
(1)安装 Visual Studio 官网：https://visualstudio.microsoft.com/zh-hans/vs/
(2)安装cmake 官网：https://cmake.org/
(3)安装boost 官网：https://www.boost.org/
(4)安装dlib 官网：http://dlib.net/

## 3.成果与分析：
整体开发环境

<img src="https://img-blog.csdnimg.cn/20191203222521442.png"   width="600px">

闭眼时EAR值下降接近0，增眼时EAR值增加

<img src="https://img-blog.csdnimg.cn/2019120322261974.png"   width="300px">   <img src="https://img-blog.csdnimg.cn/20191203222637479.png"   width="300px">

<img src="https://img-blog.csdnimg.cn/20191203222758926.png"   width="300px">  <img src="https://img-blog.csdnimg.cn/20191203222815316.png"   width="300px">

以上是一个完整的眨眼过程，当闭眼时曲线图下降至0.15以下，为了提高准确率可以设置为0.15+以下为闭眼状态，而睁眼状态可以设置为0.27以内。在戴眼镜和没戴眼镜时可以看出EAR值波动比较大，可以在判断语句模块进行编写使准确率提升。

<img src="https://img-blog.csdnimg.cn/20191203222916863.png"   width="400px">

数据分析:如上图所示前两个下降EAR为普通眨眼数据，后三个下降EAR为闭眼超时数据是可能会发生事故的闭眼时间。三个下降EAR期间反馈了一次超时次数和一次警告（两次超时次数累加为一次）。
功能实现:

```python
		if ear < EYE_EAR: # 如果EAR小于阈值，开始累计连续眨眼次数
			frame_counter += 1
		else:
			if frame_counter >= EYE_EAR_BEYOND_TIME:# 连续帧计数超过EYE_EAR_BEYOND_TIME的帧数时，累加超时次数（blink_counter+1）并给予提示警告
				winsound.Beep(freq, duration)# 提示报警声
				blink_counter += 1
				frame_counter = 0
			if blink_counter >= EYE_EAR_WARNIG:# 连续帧计数超过EYE_EAR_WARNIG帧数时，给警告次数（WARNIG+1）并给予长时间报警警告
				winsound.Beep(freq, duration2)# 长时间报警声
				WARNIG += 1
				blink_counter = 0
```

1.运用python的if语句进行EAR判断是否眨眼，然后累加眨眼帧数
2.在累加眨眼帧数期间，来判断眨眼帧数超过我们设定的超时数时，给予短暂提示声并累加一次超时次数。
3.如果累加次数超过我设定的次数时，给予长时间的警告声提示并累加警告次数。

 <img src="https://img-blog.csdnimg.cn/20191203223537688.png"   width="400px">   <img src="https://img-blog.csdnimg.cn/20191203223552452.png"   width="400px">

根据上面的测试实验可以保存EAR值和左右眼的EAR值有利于后期数据的分析与处理，并且可以根据数据绘制各类图表。

## 四、结论
我们在代码上增加了判断语句、matplotlib的实时在线绘制折线图、报警提示声、保存数据并且可以绘制各种图表、使应用效果更加完善。
不足之处有：
1.此程序对摄像头帧数有一定要求，像素越低处理越快但也要一定的分辨率使识别度提高。
2.在运行过程中识别有时会卡顿，毕竟每一帧要被处理多次，处理也需要时间，于是我们对此进行了优化：(1)提高硬件性能(2)把眼眶绘制、matplotlib绘图和保存数据模块等附加功能备注掉，在需要时打开。
3.在遮住一只眼睛的时候，出现遮住的一只眼在继续识别所发生错误，所以需要把检测点分开来，分别对单只眼睛特征点检测，所以之后还需要更进一步成熟的优化。
与其他人研究不同之处，在于这个方法与传统的计算眨眼图像处理方法是不同的，它使用眼睛的长宽比，简单的解决了人眼识别的复杂性，它基于眼睛的特征点之间的距离比例是一个非常简单的计算方法。

## 五、参考文献
[1]Soukupová和Čech“Real-Time Eye Blink Detection using Facial Landmarks”
[2]python dlib学习（十一）：眨眼检测 ,hongbin_xu,2018-01-11
[3]Eye blink detection with OpenCV, Python, and dlib,Adrian Rosebrock
[4]Win10环境python3.7安装dlib模块，Book_bei，2019-04-04
[5]Python读取 txt文档并绘制折线图,無負今日,2019-09-08 
[6]Python、opencv安装教程与环境搭建
[7]Pip安装教程
