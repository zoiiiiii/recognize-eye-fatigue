import numpy as np 
import cv2
import dlib
from scipy.spatial import distance
import os
from imutils import face_utils
import matplotlib.pyplot as plt
import winsound

def eye_aspect_ratio(eye):# 计算EAR
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

pwd = os.getcwd()# 获取当前路径
model_path = os.path.join(pwd, 'model')# 模型文件夹路径
shape_detector_path = os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat')# 人脸特征点检测模型路径
detector = dlib.get_frontal_face_detector()# 人脸检测器 
predictor = dlib.shape_predictor(shape_detector_path)# 人脸特征点检测器

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
plt.ion()# 开启一个画图窗口

leftEAR_TXT = open("eardata/leftEAR.txt","w")# 打开保存data的文本
rightEAR_TXT = open("eardata/rightEAR.txt","w")
EAR_TXT = open("eardata/ear.txt","w")

cap = cv2.VideoCapture('C:/Users/kzero/Pictures/Camera Roll/WIN_20191203_21_11_27_Pro.mp4')# 导入视频流或
#'C:/Users/kzero/Pictures/Camera Roll/WIN_20191104_11_13_31_Pro.mp4'
#改0为本地摄像头导入视频流
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

		print('{0}'.format(leftEAR),file=leftEAR_TXT)# 保存左右眼ear和ear的数据
		print('{0}'.format(rightEAR),file=rightEAR_TXT)
		print('{0}'.format(ear),file=EAR_TXT)
		
		

	if cv2.waitKey(1) & 0xFF == 27:# 按esc退出
		break
		plt.ioff() # 关闭画图的窗口

cap.release()
cv2.destroyAllWindows()

