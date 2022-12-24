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
	// 定义训练参数
	auto Train_Data = nc::vstack({ Class_1, Class_2 });
	auto m = Train_Data.shape();
	nc::NdArray<float> Y = { 1,1,1,1,1,0,0,0,0,0 };
	//auto w = nc::random::randN<float>({ 3,1 });
	nc::NdArray<float> w = {1,1};
	w = w.reshape(2, 1);
	nc::NdArray<float> b = { 1 };
	float a = 0.01;
	int epoch = 0;
	// 开始训练
	while (true)
	{
		epoch += 1;
		for (int i = 0; i < m.rows; i++)
		{
			// 这里与python不同numcpp进行向量相乘后得出的是一个1*1的矩阵，必须先用at()将数据提取出来后再进行运算
			auto fx = nc::matmul(w.transpose(), Train_Data(i, Train_Data.cSlice()).transpose()) + b;
			auto gz = 1 / (1 + nc::exp(-fx).at(0));
			// 开始梯度下降
			w[0] -= a * Train_Data(i, 0) * (gz - Y[i]);
			w[1] -= a * Train_Data(i, 1) * (gz - Y[i]);
			b -= a * (gz - Y[i]);
		}
		auto fx = nc::matmul(w.transpose(), Train_Data.transpose()) + b.at(0);
		auto gz = 1.0f / (1.0f + nc::exp(-fx));
		auto Loss = (nc::matmul(-Y, nc::log(gz).transpose()) - nc::matmul((1.0f - Y), nc::log(1.0f - gz).transpose())) / (float)m.rows;
		cout << Loss << endl;
		if (Loss.at(0) <= 0.3)
		{
			break;
		}
	}
	cout << "w的值为 \n" << w << endl;  // 输出训练结果
	cout << "b的值为 \n" << b << endl;
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