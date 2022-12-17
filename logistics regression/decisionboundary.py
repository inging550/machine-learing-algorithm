import matplotlib.pyplot as plt
import torch
import numpy as np


# 进行归一化
def normalization(Data1, Data2, Data3):
    # 要对全部的X坐标单独归一化，全部的Y坐标同理
    Data_x = np.concatenate((Data1[:, 0], Data2[:, 0], Data3[:, 0]))
    Data_y = np.concatenate((Data1[:, 1], Data2[:, 1], Data3[:, 1]))
    # 参照Matlab的mapminmax函数进行归一化
    Data_x = (Data_x - np.min(Data_x)) / (np.max(Data_x) - np.min(Data_x))
    Data_y = (Data_y - np.min(Data_y)) / (np.max(Data_y) - np.min(Data_y))
    Data_x = np.expand_dims(Data_x, 1)
    Data_y = np.expand_dims(Data_y, 1)

    Data1 = np.concatenate((Data_x[:5], Data_y[:5]), 1)
    Data2 = np.concatenate((Data_x[5:10], Data_y[5:10]), 1)
    Data3 = np.concatenate((Data_x[10:], Data_y[10:]), 1)
    return Data1, Data2, Data3


if __name__ == "__main__":
    # 定义数据集
    class_1_o = np.array([[220, 90], [240, 95], [220, 95], [180, 95], [140, 90]], dtype=np.float)
    class_2_o = np.array([[80, 85], [85, 80], [85, 85], [82, 80], [78, 80]], dtype=np.float)
    Test_Data_o = np.array([[180, 90], [210, 90], [140, 90], [90, 80], [78, 80]], dtype=np.float)
    Class_1, Class_2, Test_Data = normalization(class_1_o, class_2_o, Test_Data_o)
    a = Class_1.shape
    b = Class_2.shape
    # 开始绘图
    plt.figure()
    plt.plot(Class_1[:, 0], Class_1[:, 1], 'r*', label='Class_1')
    plt.plot(Class_2[:, 0], Class_2[:, 1], 'b*', label='Class_2')
    plt.plot(Test_Data[:, 0], Test_Data[:, 1], 'gs', label='Test_Data')
    plt.legend(loc='best')
    # 增广向量
    att_1 = np.ones((a[0], 1))
    att_2 = np.ones((b[0], 1))
    # 这里对数据集做了预处理使得 当正确分类是w * x > 0   ，错误分类是 w * x <= =
    Train_Data = np.concatenate((np.concatenate((Class_1, att_1),  axis=1), -1 * np.concatenate((Class_2, att_2),  axis=1) ), axis=0)
    Test_Data = np.concatenate((Test_Data, np.ones((5, 1))), axis=1)
    c = Train_Data.shape
    # w = np.random.randn(3, 1)
    w = np.array([[1], [1], [1]])  # 使用[1;1;1]方便与C++的运行结果比对（因为C++无法使用matlpotlib绘图）
    a = 0.1
    # 开始训练
    while True:
        bb = w
        for i in range(0, c[0]):
            if np.matmul(w.T, Train_Data[i, :].T) <= 0:  # matmul为矩阵乘法（线代知识）
                w = w + a * Train_Data[i, :].reshape(3, 1)
        if (bb == w).all():
            break
    print(w)
    # 绘制分类线
    xlist = np.linspace(-0.5, 1.5, 100)
    ylist = np.linspace(-0.5, 1.5, 100)
    x, y = np.meshgrid(xlist, ylist)  # 计算圆所在区域的网格
    f = w[0] * x + w[1] * y + w[2] < 0
    plt.contourf(x, y, f, cmap="cool")
    plt.show()


