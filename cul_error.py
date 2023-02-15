from sklearn.metrics import *
import math

def cal_error(file):
    with open(file) as files:
        f=files.readlines()
        # print(f)
        true_value=[]
        pre_value=[]
        for i,t in enumerate(f):
            true_value.append(t.strip().split(" ")[0])
            pre_value.append(t.strip().split(" ")[1])
            # print(i)
            # print(t)
            # if(i%2==0):
            #     float_i = float(t.strip())
            #     true_value.append(float_i)  # 测试集的实际值
            # else:
            #     float_j = float(t.strip())
            #     pre_value.append(float_j)
        # print(true_value)
        # print(pre_value)
        print('MAE='+str(mean_absolute_error(true_value,pre_value)))
        print('MSE='+str(mean_squared_error(true_value,pre_value)))
        print('RMSE=' + str(math.sqrt(mean_squared_error(true_value, pre_value))))
        print('r2='+str(r2_score(true_value,pre_value,sample_weight=None,multioutput='uniform_average')))

cal_error('train.dat')
print("="*25)
cal_error('test.dat')
