# Load libraries

import csv
import random
import math
import matplotlib

import numpy as np

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class IrisData():
 def __init__(self, filename):
  with open(filename, "r") as f_input:
   csv_input = csv.reader(f_input)
   self.details = list(csv_input)

 def get_col_row(self, col, row):
  return self.details[row-1][col-1] 
  # Python index starts from 0 so we have to substract by 1

data = IrisData("irisk2.csv")


sepall=[]
sepalw=[]
petall=[]
petalw=[]
category1=[]
category2=[]


for x in range(0,149):
	sepall.append(float(data.get_col_row(1,x)))
	sepalw.append(float(data.get_col_row(2,x)))
	petall.append(float(data.get_col_row(3,x)))
	petalw.append(float(data.get_col_row(4,x)))

	category1.append(float(data.get_col_row(6,x)))
	category2.append(float(data.get_col_row(7,x)))




target1=0
target2=0
sigmoid1=0
sigmoid2=0
eror1=0
eror2=0
lrate=0.1
totalerror1V=0
totalerror2V=0
accurateV=0
totalerror1L=0
totalerror2L=0
accurateL=0

sumaccurateL=[]
sumerorL=[]
sumaccurateV=[]
sumerorV=[]


sumaccurateLL=[]
sumaccurateVV=[]
sumerorLL=[]
sumerorVV=[]


for z in range (0,5):
	print('K-',z)
	x1=z*30
	x2=((z+1)*30)-1
    

	

	the1=random.uniform(0,1)
	the2=random.uniform(0,1)
	the3=random.uniform(0,1)
	the4=random.uniform(0,1)
	bias1=random.uniform(0,1)

	the5=random.uniform(0,1)
	the6=random.uniform(0,1)
	the7=random.uniform(0,1)
	the8=random.uniform(0,1)
	bias2=random.uniform(0,1)
	


	for y in range (0,100):
		print('epoch-',y)

		
		for x in range (0,x1):


			target1=float(the1*sepall[x]+the2*sepalw[x]+the3*petall[x]+the4*petalw[x]+bias1)
			target2=float(the5*sepall[x]+the6*sepalw[x]+the7*petall[x]+the8*petalw[x]+bias2)

			sigmoid1=float(1/(1+math.exp(-target1)))
			sigmoid2=float(1/(1+math.exp(-target2)))


			if sigmoid1>0.5:
				prediction1=1.0
			else:
				prediction1=0.0

			if sigmoid2>0.5:
				prediction2=1.0
			else:
				prediction2=0.0
				
			if (category1[x] == prediction1 and category2[x] == prediction2):
				accurateL=accurateL+1
			
			
			eror1=float((abs(sigmoid1-category1[x]))**2)
			eror2=float((abs(sigmoid2-category2[x]))**2)

			totalerror1L=totalerror1L+eror1
			totalerror2L=totalerror2L+eror2

			dthe1=2*(sigmoid1-category1[x])*(1-sigmoid1)*sigmoid1*sepall[x]
			dthe2=2*(sigmoid1-category1[x])*(1-sigmoid1)*sigmoid1*sepalw[x]
			dthe3=2*(sigmoid1-category1[x])*(1-sigmoid1)*sigmoid1*petall[x]
			dthe4=2*(sigmoid1-category1[x])*(1-sigmoid1)*sigmoid1*petalw[x]
			dbias1=2*(sigmoid1-category1[x])*(1-sigmoid1)*sigmoid1*1
			
			dthe5=2*(sigmoid2-category2[x])*(1-sigmoid2)*sigmoid2*sepall[x]
			dthe6=2*(sigmoid2-category2[x])*(1-sigmoid2)*sigmoid2*sepalw[x]
			dthe7=2*(sigmoid2-category2[x])*(1-sigmoid2)*sigmoid2*petall[x]
			dthe8=2*(sigmoid2-category2[x])*(1-sigmoid2)*sigmoid2*petalw[x]
			dbias2=2*(sigmoid2-category2[x])*(1-sigmoid2)*sigmoid2*1

			the1=the1-lrate*dthe1
			the2=the2-lrate*dthe2
			the3=the3-lrate*dthe3
			the4=the4-lrate*dthe4
			bias1=bias1-lrate*bias1

			the5=the5-lrate*dthe5
			the6=the6-lrate*dthe6
			the7=the7-lrate*dthe7
			the8=the8-lrate*dthe8
			bias2=bias2-lrate*bias2
			
		for x in range (x2,149):


			target1=float(the1*sepall[x]+the2*sepalw[x]+the3*petall[x]+the4*petalw[x]+bias1)
			target2=float(the5*sepall[x]+the6*sepalw[x]+the7*petall[x]+the8*petalw[x]+bias2)

			sigmoid1=float(1/(1+math.exp(-target1)))
			sigmoid2=float(1/(1+math.exp(-target2)))


			if sigmoid1>0.5:
				prediction1=1.0
			else:
				prediction1=0.0

			if sigmoid2>0.5:
				prediction2=1.0
			else:
				prediction2=0.0
				
			if (category1[x] == prediction1 and category2[x] == prediction2):
				accurateL=accurateL+1
			
			
			eror1=float((abs(sigmoid1-category1[x]))**2)
			eror2=float((abs(sigmoid2-category2[x]))**2)

			totalerror1L=totalerror1L+eror1
			totalerror2L=totalerror2L+eror2

			dthe1=2*(sigmoid1-category1[x])*(1-sigmoid1)*sigmoid1*sepall[x]
			dthe2=2*(sigmoid1-category1[x])*(1-sigmoid1)*sigmoid1*sepalw[x]
			dthe3=2*(sigmoid1-category1[x])*(1-sigmoid1)*sigmoid1*petall[x]
			dthe4=2*(sigmoid1-category1[x])*(1-sigmoid1)*sigmoid1*petalw[x]
			dbias1=2*(sigmoid1-category1[x])*(1-sigmoid1)*sigmoid1*1
			
			dthe5=2*(sigmoid2-category2[x])*(1-sigmoid2)*sigmoid2*sepall[x]
			dthe6=2*(sigmoid2-category2[x])*(1-sigmoid2)*sigmoid2*sepalw[x]
			dthe7=2*(sigmoid2-category2[x])*(1-sigmoid2)*sigmoid2*petall[x]
			dthe8=2*(sigmoid2-category2[x])*(1-sigmoid2)*sigmoid2*petalw[x]
			dbias2=2*(sigmoid2-category2[x])*(1-sigmoid2)*sigmoid2*1

			the1=the1-lrate*dthe1
			the2=the2-lrate*dthe2
			the3=the3-lrate*dthe3
			the4=the4-lrate*dthe4
			bias1=bias1-lrate*bias1

			the5=the5-lrate*dthe5
			the6=the6-lrate*dthe6
			the7=the7-lrate*dthe7
			the8=the8-lrate*dthe8
			bias2=bias2-lrate*bias2

		for x in range(x1,x2):

			target1=float(the1*sepall[x]+the2*sepalw[x]+the3*petall[x]+the4*petalw[x]+bias1)
			target2=float(the5*sepall[x]+the6*sepalw[x]+the7*petall[x]+the8*petalw[x]+bias2)

			sigmoid1=float(1/(1+math.exp(-target1)))
			sigmoid2=float(1/(1+math.exp(-target2)))


			if sigmoid1>0.5:
				prediction1=1.0
			else:
				prediction1=0.0

			if sigmoid2>0.5:
				prediction2=1.0
			else:
				prediction2=0.0
				
			if (category1[x] == prediction1 and category2[x] == prediction2):
				accurateV=accurateV+1
			
			
			eror1=float((abs(sigmoid1-category1[x]))**2)
			eror2=float((abs(sigmoid2-category2[x]))**2)

			totalerror1V=totalerror1V+eror1
			totalerror2V=totalerror2V+eror2




		erorav1=totalerror1L/120
		erorav2=totalerror2L/120
		erorav3L=erorav1+erorav2
		accuavL=accurateL/120
		

		erorav1=totalerror1V/30
		erorav2=totalerror2V/30
		erorav3V=erorav1+erorav2
		accuavV=accurateV/30
		

		sumaccurateL.append(accuavL)
		sumaccurateV.append(accuavV)
		sumerorL.append(erorav3L)
		sumerorV.append(erorav3V)

		sumaccurateLL.append(accuavL)
		sumaccurateVV.append(accuavV)
		sumerorLL.append(erorav3L)
		sumerorVV.append(erorav3V)

		totalerror1L=0
		totalerror2L=0
		accurateL=0
		totalerror1V=0
		totalerror2V=0
		accurateV=0

		
		print('Eror1 : ',erorav3L)
		print('Accuracy : ',accuavL)

		print('Eror1 : ',erorav3V)
		print('Accuracy : ',accuavV)

		y=y+1

	xaxis = np.linspace(0,100,100)



	plt.figure('Eror')
	print("COK",len(sumaccurateL))
	plt.plot(xaxis,sumaccurateL)
	plt.figure('Accurate')
	plt.plot(xaxis,sumaccurateV)

	plt.figure('Eror')
	plt.plot(xaxis,sumerorL)
	plt.figure('Accurate')
	plt.plot(xaxis,sumerorV)

	plt.show()
	z+=1

	
	sumaccurateL=[]
	sumaccurateV=[]
	sumerorL=[]
	sumerorV=[]

	

for i in range (0,99):


	sumaccurateLL[i]= (sumaccurateLL[i] +sumaccurateLL[i+100] +sumaccurateLL[i+200] + sumaccurateLL[i+300] + sumaccurateLL[i+400])/5
	sumaccurateVV[i]= (sumaccurateVV[i]	+sumaccurateVV[i+100] +sumaccurateVV[i+200] + sumaccurateVV[i+300] + sumaccurateVV[i+400])/5
	sumerorLL[i]= (sumerorLL[i] + sumerorLL[i+100] + sumerorLL[i+200] + sumerorLL[i+300] + sumerorLL[i+400])/5
	sumerorVV[i]= (sumerorVV[i] + sumerorVV[i+100] + sumerorVV[i+200] + sumerorVV[i+300] + sumerorVV[i+400])/5

	i+=1

sumaccurateLL = sumaccurateLL[:100]
sumaccurateVV = sumaccurateVV[:100]
sumerorLL = sumerorLL[:100]
sumerorVV =  sumerorVV[:100]



plt.figure('Average Eror')
print("COK",len(sumaccurateLL))
plt.plot(xaxis,sumaccurateLL)
plt.figure('Average Accurate')
plt.plot(xaxis,sumaccurateVV)

plt.figure('Average Eror')
plt.plot(xaxis,sumerorLL)
plt.figure('Average Accurate')
plt.plot(xaxis,sumerorVV)

plt.show()

