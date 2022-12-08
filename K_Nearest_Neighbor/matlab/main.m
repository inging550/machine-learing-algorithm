clc;clear all;close all   % ����
%% ʹ��randn����������ֵΪ0�������^2 = 1����׼���=1����̬�ֲ����������2000�����ݣ������ࡣ
% �ƶ�ѵ���� 
% ʹ��randn����������ֵΪ0�������^2 = 1����׼���=1����̬�ֲ����������2000�����ݣ������ࡣ
%% ʹ��randn����������ֵΪ0�������^2 = 1����׼���=1����̬�ֲ����������2000�����ݣ������ࡣ
X = [randn(5000,2)+ones(5000,2);randn(5000,2)-ones(5000,2)];
X(1:5000,3)=1;
X(5001:10000,3)=2;
% ����ѵ�����Լ����Լ�
% �������һ 1-4000Ϊѵ���� 4001��5000Ϊ���Լ�
% �������� 5001��9000Ϊѵ���� 9001��10000Ϊ���Լ� 
class1_train = X(1:4000,:);
class1_test = X(4001:5000,:);
class2_train = X(5001:9000,:);
class2_test = X(9001:10000,:);
%% ��ͼ
figure
plot(class1_train(:,1),class1_train(:,2),'r*'); hold on;
plot(class2_train(:,1),class2_train(:,2),'b*');
legend({'class1_train','class2_train'},'Location','northwest')
%% �洢
%save ('NN_2000.mat','X');
%frame = getframe(fig); % ��ȡframe
%img = frame2im(frame); % ��frame�任��imwrite��������ʶ��ĸ�ʽ
%imwrite(img,'a.png');
%% ��ʼ����K���� K��1��101����Ϊ2
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
%% ͳ�ƽ��
figure
x = linspace(1,101,51);
plot(x,1-total_Kcorrect1/1000,'r','LineWidth',2)
hold on
plot(x,1-total_Kcorrect2/1000,'g','LineWidth',2)
legend({'���һ������','����������'})