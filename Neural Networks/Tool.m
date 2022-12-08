close all;
clear;
clc
%% y=(x-1)^2  每隔0.1取一个值  在【0，2】范围内   一共21个值
x = 0:0.1:2;
y = (x-1).^2;
% 下面的数据集必须归一化  sigmoid的特性导致梯度消失  ReLu会发生梯度爆炸
% x = [9 10 11 12 13 14 15 16 17 18];
% y = [0.50 9.36 52.0 192.0 350.0 571.0 912.0 1207.0 1682.69 2135.0];  
figure;
plot(x,y,'b-');hold on;
% [x_min,Xs]=mapminmax(x,0,1);
% [y_min,Ys]=mapminmax(y,0,1);
% [y_max,Ys1]=mapminmax(y_min,0,1);
[m,n] = size(x);
%% 初始化训练参数
a = 0.01;
w = randn([1 6]);
b = randn([1 4]);
% w=[-0.4 -0.3 1.0 3.6 0.4 1];%网络连接权重
% b=[0.4 -0.6 -0.5 -0.1];%神经元偏置
epoch = 200000;
%% 开始训练
for i=1:epoch
    for j=1:n  % n = 21
        % 前向传播
        % 输入层
        s1 = x(j);
        % 隐藏层
        s2 = w(1)*s1 + b(1);
        s3 = w(2)*s1 + b(2);
        s4 = w(3)*s1 + b(3);
        if s2<0
            s2 = 0;
        end
        if s3<0
            s3 = 0;
        end
        if s4<0
            s4 = 0;
        end
        % 输出层
        s5 = w(4)*s2 + w(5)*s3 + w(6)*s4 + b(4);
        if s5<0
            s5 = 0;
        end
        %输出层和隐含层神经元的误差计算如下
        e3 = -(y(j)-s5);       
        e2 = w(6)*e3;
        e1 = w(5)*e3;
        e0 = w(4)*e3;
        % 反向传播
        % 输出层
        w(6) = w(6) - a*s4*e3;
        w(5) = w(5) - a*s3*e3;
        w(4) = w(4) - a*s2*e3;
        b(4) = b(4) - a*e3;
        % 隐藏层
        w(3)=w(3) - a*e2*s1;
        w(2)=w(2) - a*e1*s1;
        w(1)=w(1) - a*e0*s1;
        b(1)=b(1) - a*e2;
        b(2)=b(2) - a*e1;
        b(3)=b(3) - a*e0;
    end
end

%% 开始预测
% 打印预测曲线
% x1 = [9 10 11 12 13 14 15 16 17 18 19 20];
% x1_min = mapminmax('apply',x1,Xs); %反归一化
x1_min = x;
% [m_2,n_2] = size(x1);
for k=1:n
    ts2 = w(1)*x1_min(k) + b(1);
    ts3 = w(2)*x1_min(k) + b(2);
    ts4 = w(3)*x1_min(k) + b(3);
    if ts2<0
       ts2 = 0;
    end
    if ts3<0
       ts3 = 0;
    end
    if ts4<0
       ts4 = 0;
    end
    ts5(k) = w(4)*ts2 + w(5)*ts3 + w(6)*ts4 + b(4);
    if ts5<0
       ts5 = 0;
    end
end
% ts5 = mapminmax('apply',ts5,Ys1);  % 反归一化
plot(x,ts5,'r-');hold on;
grid on;
legend('训练集','训练结果')

%% 输入为1.29求输出
x_in = 1.29;
ts2 = w(1)*x_in + b(1);
ts3 = w(2)*x_in + b(2);
ts4 = w(3)*x_in + b(3);
if ts2<0
   ts2 = 0;
end
if ts3<0
   ts3 = 0;
end
if ts4<0
   ts4 = 0;
end
ts5 = w(4)*ts2 + w(5)*ts3 + w(6)*ts4 + b(4);
fprintf('若输入为1.29输出为%f\n',ts5)