clc;
close all;
clear all;
%% ѵ������
Class_1=  [0 0;1 0];
Class_2=  [0 1;1 1];
[m_1,n_1] = size(Class_1);   % m_1 ��ʾ�ж��ٸ�����
[m_2,n_2] = size(Class_2);
%% ��ͼ����
figure;
plot(Class_1(:,1),Class_1(:,2),'*');hold on;
plot(Class_2(:,1),Class_2(:,2),'o');hold on;
%%  ��������
att_1=ones(m_1,1);
att_2=ones(m_2,1);
Train_Data = [Class_1,att_1;Class_2,att_2];
epoch = 0;
a = 0.1;
w = randn([3,1]); %�����ʼ��Ȩֵ
%% ʹ��������ʽ����Ȩֵ������
while 1==1
    epoch = epoch + 1;
    %a = 1/epoch;
    for i=1:m_1  % �������1
        output = w'*Train_Data(i,:)';
        fx=1/(1+exp(-output));    % sigmoid����
        if fx < 0.5  % ��ʾ�������  ���1Ӧ��������1
            w = w + a*Train_Data(i,:)'*(1-fx)*fx*(1-fx);
        end
    end
    for i=(m_1+1):(m_1+m_2)  % �������2
        output = w'*Train_Data(i,:)';
        fx=1/(1+exp(-output));
        if fx >= 0.5 % ������� ���2Ӧ������0
            w = w + a*Train_Data(i,:)'*(-fx)*fx*(1-fx);
        end        
    end
    % �����ʱ�ﵽҪ��(ֹͣ����)
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
%% ��ӡ���
f=[num2str(wo(1,1)) '*x+(' num2str(wo(2,1)) ')*y+(' num2str(wo(3,1)) ')'];
h = ezplot(f,[-0.5,1.5]);
grid on;
set(h,'Color','r');
legend('class1','class2','decisionboundary')

for i=1:(m_1+m_2)
    output = wo'*Train_Data(i,:)';
    fx=1/(1+exp(-output));
    if fx>0.5
        fprintf("��%d���������ڵ�1�����\n",i);
    elseif fx<0.5
        fprintf("��%d���������ڵڶ������\n",i);
    end
end
