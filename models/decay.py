import matplotlib.pyplot as plt
import numpy as np
def xx(x):
    return 1.0/(1+np.log2(x+0.02)/np.log2(10000))
xxxxx=[]
for i in range(1,60):
    xxxxx.append(86123 * xx(i))
print xxxxx[0],xxxxx[-1]
# plt.plot(xxxxx,label="ypred")
# plt.show()
# yyy=[]
# for i in range(0,10):
# 	tmp=xxxxx[i]+(np.random.random_sample()-0.5)*.1
# 	plt.plot(i,tmp,"o",color="blue")
# 	yyy.append(tmp)
# y=np.array(xxxxx)
# y=y-(xxxxx[9]-yyy[9])
# plt.plot(y,label="yfixed")
# plt.legend()
# plt.show()