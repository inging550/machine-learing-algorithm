close all;
clear;
clc
%% y=(x-1)^2  每隔0.1取一个值  在【0，2】范围内   一共21个值
% x = 0:0.1:2;
% y = (x-1).^2;
%% 下面的数据集必须归一化 否则发生梯度消失   sigmoid的特性导致
x = [9 10 11 12 13 14 15 16 17 18];
y = [0.50 9.36 52.0 192.0 350.0 571.0 912.0 1207.0 1682.69 2135.0];  
figure;
plot(x,y,'b-');hold on;
%% 归一化（只有第二个数据集需要）
[x_min,Xs]=mapminmax(x,0,0.5);
[y_min,Ys]=mapminmax(y,0,0.5);
[y_max,Ys1]=mapminmax(y_min,0.5,2135.0);
[m,n] = size(x);
%% 初始化训练参数
a = 0.01;
w = randn([1 6]);
b = randn([1 4]);
epoch = 1000000;
err_limit = 0.0001;   %%误差平方和限制
%% 开始训练
for i=1:epoch
    err=0;
    for j=1:n  % n = 21
        % 前向传播
        % 输入层
        o1 = x_min(j);
        % 隐藏层
        s2 = w(1)*o1 + b(1);
        s3 = w(2)*o1 + b(2);
        s4 = w(3)*o1 + b(3);
        o2 = 1/(1+exp(-s2));
        o3 = 1/(1+exp(-s3));
        o4 = 1/(1+exp(-s4));
        % 输出层
        s5 = w(4)*o2 + w(5)*o3 + w(6)*o4 + b(4);
        o5 = 1/(1+exp(-s5));
        %输出层和隐含层神经元的误差计算如下
        e3 = o5*(1-o5)*(y_min(j)-o5);
        e2 = e3*w(6)*o4*(1-o4);
        e1 = e3*w(5)*o3*(1-o3);
        e0 = e3*w(4)*o2*(1-o2);
        % 反向传播
        % 输出层
        w(6) = w(6) + a*o4*e3;
        w(5) = w(5) + a*o3*e3;
        w(4) = w(4) + a*o2*e3;
        b(4) = b(4) + a*e3;
        % 隐藏层
        w(3) = w(3) + a*e2*o1;
        w(2) = w(2) + a*e1*o1;
        w(1) = w(1) + a*e0*o1;
        b(1) = b(1) + a*e2;
        b(2) = b(2) + a*e1;
        b(3) = b(3) + a*e0;
        err = err+(y_min(j)-o5)^2;
    end
    if err<err_limit   % 若损失较小则提前结束迭代
        break;
    end
end
%% 开始预测
% 打印预测曲线
x1 = [9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24];
x1_min = mapminmax('apply',x1,Xs); %反归一化
[m_2,n_2] = size(x1);
for k=1:n_2
    ts2 = w(1)*x1_min(k) + b(1);
    ts3 = w(2)*x1_min(k) + b(2);
    ts4 = w(3)*x1_min(k) + b(3);
    to2 = 1/(1+exp(-ts2));
    to3 = 1/(1+exp(-ts3));
    to4 = 1/(1+exp(-ts4));
    ts5 = w(4)*to2 + w(5)*to3 + w(6)*to4 + b(4);
    to5(k) = 1/(1+exp(-ts5));
end
to5 = mapminmax('apply',to5,Ys1);  % 反归一化
plot(x1,to5,'r-');hold on;
grid on;
legend('训练集','训练结果')
set(legend,'Location','NorthWest');