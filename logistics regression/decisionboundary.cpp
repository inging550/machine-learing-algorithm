#include"NumCpp.hpp"
#include"boost/filesystem.hpp"   // 这一行貌似可以去掉
#include<iostream>
using namespace std;

tuple<nc::NdArray<float>, nc::NdArray<float>, nc::NdArray<float>> normalization(nc::NdArray<float> Data1, nc::NdArray<float> Data2, nc::NdArray<float> Data3);


void main()
{
	// 定义数据集
	nc::NdArray<float> class_1_o = { {220,90},{240,95},{220,95},{180,95},{140,90} };
	nc::NdArray<float> class_2_o = { {80,85},{85,80},{85,85},{82,80},{78,80} };
	nc::NdArray<float> Test_Data_o = { {180,90},{210,90},{140,90},{90,80},{78,80} };
	// 进行归一化Data是一个元组
	auto Data = normalization(class_1_o, class_2_o, Test_Data_o); 
	auto Class_1 = get<0>(Data);
	auto Class_2 = get<1>(Data);
	auto Test_Data = get<2>(Data);
	// 增广向量
	auto att_1 = nc::ones<float>({5,1});
	auto Train_Data = nc::vstack({ nc::hstack({ Class_1, att_1 }), -1.0f * nc::hstack({Class_2,att_1}) });
	Test_Data = nc::hstack({ Test_Data,att_1 });
	// 定义训练的参数
	//auto w = nc::random::randN<float>({ 3,1 });
	nc::NdArray<float> w = {1,1,1};
	w = w.reshape(3, 1);
	float a = 0.1;
	// 开始训练
	while (true)
	{
		auto bb = w;
		for (int i = 0; i < 10; i++)
		{
			// 这里与python不同numcpp进行向量相乘后得出的是一个1*1的矩阵，必须先用at()将数据提取出来后再进行if判断
			if (nc::matmul(w.transpose(), Train_Data(i, Train_Data.cSlice()).transpose()).at(0) <= 0)
			{
				w = w + a * Train_Data(i, Train_Data.cSlice()).reshape(3, 1);
			}
		}
		// 同理运行 bb == w 后得到的是一个1*1的矩阵，要用at()将数据提取出来
		if ((bb == w).at(0))
		{
			break;
		}
	}
	cout << w << endl;  // 输出训练结果
}


// 归一化函数
tuple<nc::NdArray<float>, nc::NdArray<float>, nc::NdArray<float>> normalization(nc::NdArray<float> Data1, nc::NdArray<float> Data2, nc::NdArray<float> Data3)
{
	//  AXis::Row相当于python的axis=0     Data1(Data1.rSlice(),0)相当于python的 Data1[:,0]
	auto Data_x = nc::concatenate({ Data1(Data1.rSlice(),0),Data2(Data2.rSlice(),0),Data3(Data3.rSlice(),0) }, nc::Axis::ROW);
	auto Data_y = nc::concatenate({ Data1(Data1.rSlice(),1),Data2(Data2.rSlice(),1),Data3(Data3.rSlice(),1) }, nc::Axis::ROW);

	// nc::min(Data_x) 返回1*1的矩阵需要用at(0)将数字提取出来再运算
	Data_x = (Data_x - nc::min(Data_x).at(0)) / (nc::max(Data_x).at(0) - nc::min(Data_x).at(0));
	Data_y = (Data_y - nc::min(Data_y).at(0)) / (nc::max(Data_y).at(0) - nc::min(Data_y).at(0));
	Data1 = nc::concatenate({ Data_x({0,5},0), Data_y({0,5},0) },nc::Axis::COL);
	Data2 = nc::concatenate({ Data_x({5,10},0), Data_y({5,10},0) }, nc::Axis::COL);
	Data3 = nc::concatenate({ Data_x({10,15},0), Data_y({10,15},0) }, nc::Axis::COL);	
	tuple<nc::NdArray<float>, nc::NdArray<float>, nc::NdArray<float>> Data(Data1,Data2,Data3);  // 定义一个元组作为返回值
	return Data;
}