close all;
clear ;
clc
%% 训练样本
Class_1=  [0 1;0 1];
Class_2=  [0 1;1 0];
[m_1,n_1] = size(Class_1);
[m_2,n_2] = size(Class_2);
Train_X=[Class_1 Class_2];
Train_Y=[1 1 0 0];  %%对应类别
l=0.7;%学习率
w=[0.2,0.3,0.4,0.1,0.3,0.2];%网络连接权重
p=[0.4,0.2,0];%神经元偏置
err_limit=0.001;
%% 开始训练
for i=1:1000000
    err=0;
    total_err=0;
    for j=1:4
        x=Train_X(:,j)';
        y=Train_Y(:,j);
        %每个神经元的总输入和输出计算如下
        o2=x(2);
        o1=x(1);
        s4=w(1)*x(1)+w(3)*x(2)+p(1);
        o4=1/(1+exp(-s4));
        s5=w(2)*x(1)+w(4)*x(2)+p(2);
        o5=1/(1+exp(-s5));
        s6=o4*w(5)+o5*w(6)+p(3);
        o6=1/(1+exp(-s6));
        err=y-o6;
        total_err=total_err+err^2;        
        %输出层和隐含层神经元的误差计算如下
        e6=o6*(1-o6)*(y-o6);
        e5=o5*(1-o5)*e6*w(6);
        e4=o4*(1-o4)*e6*w(5);       
        %网络连接权值和神经元偏置更新如下
        w(6)=w(6)+l*o5*e6;
        w(5)=w(5)+l*o4*e6;
        w(4)=w(4)+l*o2*e5;
        w(3)=w(3)+l*o2*e4;
        w(2)=w(2)+l*o1*e5;
        w(1)=w(1)+l*o1*e4;
        p(1)=p(1)+l*e4;
        p(2)=p(2)+l*e5;
        p(3)=p(3)+l*e6;
    end
    if total_err<err_limit
        break
    end
end
%% 开始测试
result = randn([1 4]);
for k=1:4
    %每个神经元的总输入和输出计算如下
    x=Train_X(:,k);
    s4=w(1)*x(1)+w(3)*x(2)+p(1);
    o4=1/(1+exp(-s4));
    s5=w(2)*x(1)+w(4)*x(2)+p(2);
    o5=1/(1+exp(-s5));
    s6=o4*w(5)+o5*w(6)+p(3);
    o6=1/(1+exp(-s6));
    % 测试样本类别
    if o6>0.5
        result(k) = 1;
        fprintf('样本%d属于第一类\n',k)
    else
        result(k) = 0;
        fprintf('样本%d属于第二类\n',k)
    end
end