clc;clear all;close all   % 清理
%% 使用randn函数产生均值为0，方差σ^2 = 1，标准差σ=1的正态分布的随机矩阵，2000个数据，分两类。
% 制定训练集 
% 使用randn函数产生均值为0，方差σ^2 = 1，标准差σ=1的正态分布的随机矩阵，2000个数据，分两类。
%% 使用randn函数产生均值为0，方差σ^2 = 1，标准差σ=1的正态分布的随机矩阵，2000个数据，分两类。
X = [randn(5000,2)+ones(5000,2);randn(5000,2)-ones(5000,2)];
X(1:5000,3)=1;
X(5001:10000,3)=2;
% 划分训练集以及测试集
% 对于类别一 1-4000为训练集 4001－5000为测试集
% 对于类别二 5001－9000为训练集 9001－10000为测试集 
class1_train = X(1:4000,:);
class1_test = X(4001:5000,:);
class2_train = X(5001:9000,:);
class2_test = X(9001:10000,:);
%% 绘图
figure
plot(class1_train(:,1),class1_train(:,2),'r*'); hold on;
plot(class2_train(:,1),class2_train(:,2),'b*');
legend({'class1_train','class2_train'},'Location','northwest')
%% 存储
%save ('NN_2000.mat','X');
%frame = getframe(fig); % 获取frame
%img = frame2im(frame); % 将frame变换成imwrite函数可以识别的格式
%imwrite(img,'a.png');
%% 开始运行K近邻 K从1到101步长为2
total_Kcorrect1 = [];
for i=1:2:101
    [num_correct1] = Loss(class1_train,class2_train,class1_test,i);
    total_Kcorrect1 = [total_Kcorrect1;num_correct1];
end

total_Kcorrect2 = [];
for j=1:2:101
    [num_correct2] = Loss(class1_train,class2_train,class2_test,j);
    total_Kcorrect2 = [total_Kcorrect2;num_correct2];
end
%% 统计结果
figure
x = linspace(1,101,51);
plot(x,1-total_Kcorrect1/1000,'r','LineWidth',2)
hold on
plot(x,1-total_Kcorrect2/1000,'g','LineWidth',2)
legend({'类别一错误率','类别二错误率'})