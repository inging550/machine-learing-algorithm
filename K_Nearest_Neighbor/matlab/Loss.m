function [correct_num] = Loss(class1,class2,test,k)
    x = size(test,1);  % test���ݼ��ж�����
    num1=0;
    num2=0;
    correct_num=0;
    distance = [];
    for i=1:x
        % ��ʼ���㵥����������������8000��ѵ�����ľ���
        distance1 = sqrt( (class1(:,1)-test(i,1)).^2 + (class1(:,2)-test(i,2)).^2 );
        %distance = [distance;distance1];
        distance2 = sqrt( (class2(:,1)-test(i,1)).^2 + (class2(:,2)-test(i,2)).^2 );
        distance = [distance1;distance2];
        % ������������ȡǰk��
        [~,index] = sortrows(distance);
        for j=1:k
           if index(j,1) < 4000
               num1 = num1 +1;
               
           elseif index(j,1) > 4000
               num2 = num2 + 1;
           end
        end
        if num1 > num2
            if test(i,3) == 1
                correct_num = correct_num + 1;
            end
        elseif num1 < num2
            if test(i,3) == 2
                correct_num = correct_num + 1;
            end
        end
        num1 = 0;
        num2 = 0;
    end
end
