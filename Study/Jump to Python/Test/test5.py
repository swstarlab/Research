#Q1
class Calculator:
	def __init__(self):
		self.value=0

	def add(self, val):
		self.value += val

class UpgradeCalculator(Calculator):
	def minus(self, val):
		self.value -= val

cal= UpgradeCalculator()
cal.add(10)
cal.minus(7)

print(cal.value)

#Q2 못 풂
class Calculator:
	def __init__(self):
		self.value=0

	def add(self, val):
		self.value += val

class MaxLimitCalculator(Calculator):
	def add(self, val):
		self.value += val
		if self.value >=100:
			self.value= 100###???return 100은 왜 안 되나?
		else:
			return self.value

cal = MaxLimitCalculator()
cal.add(50)
cal.add(40)

print(cal.value)

#Q3
all([1,2,abs(-3)-3])
#False, 왜냐하면, abs(-3)은 절댓값이므로 3이 되고,
#뒤의 -3에 의해 0이 되는데, 0은 False이므로 False가 출력된다.

chr(ord('a'))=='a'
#True, ord(a)는 a의 유니코드를 출력하고,
#chr은 다시 해당 유니코드의 문자값을 반환하기에 'a'와 같아진다.

#Q4
a=[1,-2,3,-5,8,-3]
pos=lambda a: a>0
print(list(filter(pos,a)))

#Q5
a=int(0xea)
print(a)
###???int('1A',16)이어야 하는데 작은 따옴표도 안 붙였고,
# 16도 안 줬는데 왜 올바른 답이 나오나?

#Q6
a=[1,2,3,4]
double=lambda a: a*2
print(list(map(double,a)))

#Q7
a=[-8,2,7,5,-3,5,0,1]
print(max(a)+min(a))

#Q8
round(17/3,4)

#Q9 ###못 풂
import sys
a=sys.argv[1:]###[1:]을 넣지 않음
sum=0
for i in a:
	sum += int(i)
print(sum)

#Q10 ###문제를 잘못 이해함
import os
os.chdir("C:\doit")
result=os.popen("dir")
print(result.read())

#Q11
import glob
glob.glob("C:\doit\*.py")

#Q12 뒤의 변수를 생략해도 되는지 몰랐음
import time
time.strftime('%Y',time.localtime(time.time()))\
+"/"+time.strftime('%m',time.localtime(time.time()))\
+"/"+time.strftime('%d',time.localtime(time.time()))\
+" "+time.strftime('%X',time.localtime(time.time()))

#Q13 다른 방법으로 풂
import random
def fortyfive():
	data=[]
	for i in range(1,46):
		data.append(i)
	return(data)

forty=fortyfive()

random.shuffle(forty)

for i in range(6):
print(forty.pop())