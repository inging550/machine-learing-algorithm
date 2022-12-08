clc;
close all;
clear all;
%% 训练样本
Class_1=  [0 0;1 0];
Class_2=  [0 1;1 1];
[m_1,n_1] = size(Class_1);   % m_1 表示有多少个样本
[m_2,n_2] = size(Class_2);
%% 绘图程序
figure;
plot(Class_1(:,1),Class_1(:,2),'*');hold on;
plot(Class_2(:,1),Class_2(:,2),'o');hold on;
%%  增广向量
att_1=ones(m_1,1);
att_2=ones(m_2,1);
Train_Data = [Class_1,att_1;Class_2,att_2];
epoch = 0;
a = 0.1;
w = randn([3,1]); %随机初始化权值
%% 使用批处理方式计算权值修正量
while 1==1
    epoch = epoch + 1;
    %a = 1/epoch;
    for i=1:m_1  % 遍历类别1
        output = w'*Train_Data(i,:)';
        fx=1/(1+exp(-output));    % sigmoid函数
        if fx < 0.5  % 表示分类错误  类别1应该趋向与1
            w = w + a*Train_Data(i,:)'*(1-fx)*fx*(1-fx);
        end
    end
    for i=(m_1+1):(m_1+m_2)  % 遍历类别2
        output = w'*Train_Data(i,:)';
        fx=1/(1+exp(-output));
        if fx >= 0.5 % 分类错误 类别2应该趋向0
            w = w + a*Train_Data(i,:)'*(-fx)*fx*(1-fx);
        end        
    end
    % 检验何时达到要求(停止迭代)
    right_num=0;
    for i=1:m_1
        output = w'*Train_Data(i,:)';
        fx=1/(1+exp(-output));
        if fx > 0.5
            right_num = right_num+1;
        else
            right_num=0;
        end
    end
    for i=(m_1+1):(m_1+m_2)
        output = w'*Train_Data(i,:)';
        fx=1/(1+exp(-output));
        if fx <= 0.5
            right_num = right_num+1;
        else
            right_num=0;
        end
    end
	if right_num>(m_1+m_2-1)
        wo=w;
        break
	end
end
%% 打印结果
f=[num2str(wo(1,1)) '*x+(' num2str(wo(2,1)) ')*y+(' num2str(wo(3,1)) ')'];
h = ezplot(f,[-0.5,1.5]);
grid on;
set(h,'Color','r');
legend('class1','class2','decisionboundary')

for i=1:(m_1+m_2)
    output = wo'*Train_Data(i,:)';
    fx=1/(1+exp(-output));
    if fx>0.5
        fprintf("第%d个样本属于第1个类别\n",i);
    elseif fx<0.5
        fprintf("第%d个样本属于第二个类别\n",i);
    end
end
