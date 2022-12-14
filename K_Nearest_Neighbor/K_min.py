import torch
import matplotlib.pyplot as plt


def Loss(class1, class2, test, k):
    x = test.shape
    correct_num = 0
    distance = torch.Tensor([]).cuda()
    # distance = torch.Tensor([])
    print(distance.device)
    for i in range(0, x[0]):
        num1 = 0
        num2 = 0
        distance1 = torch.sqrt( torch.square(class1[:, 0]-test[i, 0]) + torch.square(class1[:, 1]-test[i, 1]) )
        distance1.to(device)
        distance2 = torch.sqrt( torch.square(class2[:, 0]-test[i, 0]) + torch.square(class2[:, 1]-test[i, 1]) )
        distance2.to(device)
        distance = torch.cat((distance1,distance2), 0)
        _, index = torch.sort(distance, 0)
        for j in range(0, k):
            if index[j] <= 4000:
                num1 = num1 + 1
            elif index[j] > 4000:
                num2 = num2 + 1
        if num1 > num2 and test[i, 2] == 1:
            correct_num = correct_num + 1
        elif num1 < num2 and test[i, 2] == 2:
            correct_num = correct_num + 1
    return correct_num

if __name__ == "__main__":
    ## 定义数据集
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    X1 = torch.randn(5000, 2) + torch.ones(5000, 2)
    X2 = torch.randn(5000, 2) - torch.ones(5000, 2)
    X = torch.cat((X1, X2), 0, )  # (10000,2)
    label = torch.IntTensor(10000, 1)
    label[:5000] = 1
    label[5000:] = 2
    Data_set = torch.cat((X, label), 1)
    # Data_set = Data_set.cuda()
    class1_train = Data_set[:4000, :]
    # print(class1_train.device)
    class1_test = Data_set[4000:5000, :]
    class2_train = Data_set[5000:9000, :]
    class2_test = Data_set[9000:, :]
    ## 绘制数据集
    plt.plot(class1_train[:, 0], class1_train[:, 1], 'r+', label="class1_Train")
    plt.plot(class2_train[:, 0], class2_train[:, 1], 'b+', label="class1_Train")
    plt.legend(loc='upper right')
    plt.show()
    ## 开始训练
    total_Kcorrect1 = []
    for num_k in range(1, 101, 2):
        num_correct = Loss(class1_train, class2_train, class1_test, num_k)
        total_Kcorrect1.append(num_correct)

    total_Kcorrect2 = []
    for num_k in range(1, 101, 2):
        num_correct = Loss(class1_train, class2_train, class2_test, num_k)
        total_Kcorrect2.append(num_correct)

    total_Kcorrect2 = list(map(lambda x : float(1-x/1000), total_Kcorrect2))
    total_Kcorrect1 = list(map(lambda x : float(1-x/1000), total_Kcorrect1))
    plt.figure()
    x = range(1, 101, 2)
    plt.plot(x, total_Kcorrect1[:], 'r-', label='Class 1: Error rate')
    plt.plot(x, total_Kcorrect2[:], 'g-', label='Class 2: Error rate')
    plt.legend()
    plt.show()
