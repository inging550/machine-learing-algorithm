#include"NumCpp.hpp"
#include"boost/filesystem.hpp"   // ��һ��ò�ƿ���ȥ��
#include<iostream>
using namespace std;

tuple<nc::NdArray<float>, nc::NdArray<float>, nc::NdArray<float>> normalization(nc::NdArray<float> Data1, nc::NdArray<float> Data2, nc::NdArray<float> Data3);


void main()
{
	// �������ݼ�
	nc::NdArray<float> class_1_o = { {220,90},{240,95},{220,95},{180,95},{140,90} };
	nc::NdArray<float> class_2_o = { {80,85},{85,80},{85,85},{82,80},{78,80} };
	nc::NdArray<float> Test_Data_o = { {180,90},{210,90},{140,90},{90,80},{78,80} };
	// ���й�һ��Data��һ��Ԫ��
	auto Data = normalization(class_1_o, class_2_o, Test_Data_o); 
	auto Class_1 = get<0>(Data);
	auto Class_2 = get<1>(Data);
	auto Test_Data = get<2>(Data);
	// ��������
	auto att_1 = nc::ones<float>({5,1});
	auto Train_Data = nc::vstack({ nc::hstack({ Class_1, att_1 }), -1.0f * nc::hstack({Class_2,att_1}) });
	Test_Data = nc::hstack({ Test_Data,att_1 });
	// ����ѵ���Ĳ���
	//auto w = nc::random::randN<float>({ 3,1 });
	nc::NdArray<float> w = {1,1,1};
	w = w.reshape(3, 1);
	float a = 0.1;
	// ��ʼѵ��
	while (true)
	{
		auto bb = w;
		for (int i = 0; i < 10; i++)
		{
			// ������python��ͬnumcpp����������˺�ó�����һ��1*1�ľ��󣬱�������at()��������ȡ�������ٽ���if�ж�
			if (nc::matmul(w.transpose(), Train_Data(i, Train_Data.cSlice()).transpose()).at(0) <= 0)
			{
				w = w + a * Train_Data(i, Train_Data.cSlice()).reshape(3, 1);
			}
		}
		// ͬ������ bb == w ��õ�����һ��1*1�ľ���Ҫ��at()��������ȡ����
		if ((bb == w).at(0))
		{
			break;
		}
	}
	cout << w << endl;  // ���ѵ�����
}


// ��һ������
tuple<nc::NdArray<float>, nc::NdArray<float>, nc::NdArray<float>> normalization(nc::NdArray<float> Data1, nc::NdArray<float> Data2, nc::NdArray<float> Data3)
{
	//  AXis::Row�൱��python��axis=0     Data1(Data1.rSlice(),0)�൱��python�� Data1[:,0]
	auto Data_x = nc::concatenate({ Data1(Data1.rSlice(),0),Data2(Data2.rSlice(),0),Data3(Data3.rSlice(),0) }, nc::Axis::ROW);
	auto Data_y = nc::concatenate({ Data1(Data1.rSlice(),1),Data2(Data2.rSlice(),1),Data3(Data3.rSlice(),1) }, nc::Axis::ROW);

	// nc::min(Data_x) ����1*1�ľ�����Ҫ��at(0)��������ȡ����������
	Data_x = (Data_x - nc::min(Data_x).at(0)) / (nc::max(Data_x).at(0) - nc::min(Data_x).at(0));
	Data_y = (Data_y - nc::min(Data_y).at(0)) / (nc::max(Data_y).at(0) - nc::min(Data_y).at(0));
	Data1 = nc::concatenate({ Data_x({0,5},0), Data_y({0,5},0) },nc::Axis::COL);
	Data2 = nc::concatenate({ Data_x({5,10},0), Data_y({5,10},0) }, nc::Axis::COL);
	Data3 = nc::concatenate({ Data_x({10,15},0), Data_y({10,15},0) }, nc::Axis::COL);	
	tuple<nc::NdArray<float>, nc::NdArray<float>, nc::NdArray<float>> Data(Data1,Data2,Data3);  // ����һ��Ԫ����Ϊ����ֵ
	return Data;
}