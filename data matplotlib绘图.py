import matplotlib.pyplot as plt
data_1 = []
data_2 = []
data_3 = [] 

file_1 = open('data/EAR.txt')  #打开文档
EAR = file_1.readlines() 
file_2 = open('data/leftEAR.txt') 
leftEAR = file_2.readlines()
file_3 = open('data/rightEAR.txt') 
rightEAR = file_3.readlines()

for num in EAR:
	data_1.append(float(num.split('\n')[0]))
for num in leftEAR:
   data_2.append(float(num.split('\n')[0]))
for num in rightEAR:
	data_3.append(float(num.split('\n')[0]))

plt.subplot(311)
plt.title('EAR')
plt.plot(data_1)

plt.subplot(312)
plt.title('leftEAR')
plt.plot(data_2)

plt.subplot(313)
plt.title('rightEAR')
plt.plot(data_3)

plt.tight_layout()
plt.show()