clc;
close all;
clear all;
%% 定义训练样本
Class_1_o=  [220 90;240 95;220 95;180 95;140 90];
Class_2_o=  [80 85;85 80;85 85;82 80;78 80];
Test_Data_o = [180 90;210 90;140 90;90 80;78 80];
% 数据集归一化
Class_x = mapminmax([Class_1_o(:,1)',Class_2_o(:,1)',Test_Data_o(:,1)'], 0, 1); 
Class_y = mapminmax([Class_1_o(:,2)',Class_2_o(:,2)',Test_Data_o(:,2)'], 0, 1);
Class_1 = [Class_x(1:5)',Class_y(1:5)'];
Class_2 = [Class_x(6:10)',Class_y(6:10)'];
Test_Data = [Class_x(11:15)',Class_y(11:15)'];
%% 绘图程序
figure;
plot(Class_1(:,1),Class_1(:,2),'r*','LineWidth',2); hold on;
plot(Class_2(:,1),Class_2(:,2),'bo','LineWidth',2); hold on;
plot(Test_Data(:,1),Test_Data(:,2),'gs','LineWidth',2); hold on;
%% 定义训练参数
Train_Data = [Class_1;Class_2];
[m_3,n_3] = size(Train_Data);
Y = [1,1,1,1,1,0,0,0,0,0]; % 定义标签值
% w = randn([2 1]);  % 随机初始化权重以及参数
% b = randn(1);
w = [1;1];
b = 1;
a = 0.01;  % 学习率
epoch = 0;   % 定义迭代次数
%% 使用批处理方式计算权值修正量
while epoch<=100
    epoch = epoch + 1;
    for i=1:m_3
        fx = w'*Train_Data(i,:)' + b;  
        gz = 1/(1+exp(-fx));
        w(1) = w(1) - a*Train_Data(i,1)'*(gz - Y(i));
        w(2) = w(2) - a*Train_Data(i,2)'*(gz - Y(i));
        b = b - a*(gz-Y(i));
    end
    % 判断何时停止
    fx = w'*Train_Data' + b;
    gz = 1./(1+exp(-fx));
    Loss = (-Y*log(gz)' - (1-Y)*log(1-gz)')/m_3;
    if Loss <= 0.3
        break;
    end
end
%% 打印结果
f=[num2str(w(1)) '*x+(' num2str(w(2)) ')*y+(' num2str(b) ')'];
h = ezplot(f,[-0.5,1.5]);
grid on;
set(h,'Color','r');
legend('class1','class2','decisionboundary')