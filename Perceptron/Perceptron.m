clc;
close all;
clear all;
%% 定义训练样本
Class_1_o=  [220 90;240 95;220 95;180 95;140 90];
Class_2_o=  [80 85;85 80;85 85;82 80;78 80];
Test_Data_o = [180 90;210 90;140 90;90 80;78 80];

Class_x = mapminmax([Class_1_o(:,1)',Class_2_o(:,1)',Test_Data_o(:,1)'], 0, 1); 
Class_y = mapminmax([Class_1_o(:,2)',Class_2_o(:,2)',Test_Data_o(:,2)'], 0, 1);
Class_1 = [Class_x(1:5)',Class_y(1:5)'];
Class_2 = [Class_x(6:10)',Class_y(6:10)'];
Test_Data = [Class_x(11:15)',Class_y(11:15)'];

[m_1,n_1] = size(Class_1);
[m_2,n_2] = size(Class_2);
%% 绘图程序
figure;
plot(Class_1(:,1),Class_1(:,2),'r*','LineWidth',2); hold on;
plot(Class_2(:,1),Class_2(:,2),'bo','LineWidth',2); hold on;
plot(Test_Data(:,1),Test_Data(:,2),'gs','LineWidth',2); hold on;
%% 增广向量
att_1=ones(m_1,1);
att_2=ones(m_2,1);
Train_Data = [Class_1,att_1; -1*Class_2,-1*att_2];
Test_Data = [Test_Data,ones(5,1)];
[m_3,n_3] = size(Train_Data);
w = randn([3 1]);  % 随机初始化权重以及参数
a = 0.1;  % 学习率
epoch = 0;   % 定义迭代次数
%% 开始训练
while 1==1
    % 找到分类错误的样本
    b=w;
     for i=1:m_3
     if w'*Train_Data(i,:)' <= 0
         w = w + a * Train_Data(i,:)';
     end
    end
    if b==w
        break;
    end
    epoch = epoch + 1;
end
%% 绘制最终分类线
f = [num2str(w(1,1)) '*x1+(' num2str(w(2,1)) ')*x2+(' num2str(w(3,1)) ')'] ; 
h = ezplot(f,[0,1]);
grid on;
set(h,'Color','r');
legend('Class1','Class2','Test','DecisionBoundary')
set(legend,'location','SouthEast')
%% 打印结果
for k=1:5
    Result = w'*Test_Data(k,:)';
    if Result > 0
        fprintf('测试样本%d结果为%f属于第一类 \n',k,Result);
    elseif Result <= 0
        fprintf('测试样本%d结果为%f属于第二类 \n',k,Result);
    end
end