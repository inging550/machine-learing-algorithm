#include"NumCpp.hpp"
#include"boost/filesystem.hpp"
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include"tinyxml2.h"
#include<iostream>
using namespace std;

torch::DeviceType device_type = torch::kCUDA;
long Loss(torch::Tensor class1, torch::Tensor class2, torch::Tensor test, int k);

void main()
{
	vector<int> total_Kcorrect1;
	vector<int> total_Kcorrect2;
	long  num_correct = 0;

	torch::Tensor X1 = torch::randn({ 5000,2 }) + torch::ones({ 5000,2 });
	torch::Tensor X2 = torch::randn({ 5000,2 }) - torch::ones({ 5000,2 });
	torch::Tensor X = torch::cat({ X1, X2 }, 0);
	torch::Tensor label1 = torch::full({ 5000,1 },1);
	torch::Tensor label2 = torch::full({ 5000,1 }, 2);
	torch::Tensor label = torch::cat({ label1, label2 }, 0);
	torch::Tensor data_set = torch::cat({ X,label }, 1).to(device_type);  // GPU   CUDA0
	// cout << data_set.device() << endl;
	// 划分训练集以及测试集
	auto class1_train = data_set.narrow(0, 0, 4000);
	auto class1_test = data_set.narrow(0, 4000, 1000);
	auto class2_train = data_set.narrow(0, 5000, 4000);
	auto class2_test = data_set.narrow(0, 9000, 1000);
	

	for (int m1 = 1; m1 <= 101; m1 = m1 + 2)
	{
		num_correct = Loss(class1_train, class2_train, class1_test, 1);
		total_Kcorrect1.push_back(num_correct);
	}
	for (int m2 = 0; m2 < 101; m2 = m2 + 2)
	{
		int num_correct = Loss(class1_train, class2_train, class1_test, 1);
		total_Kcorrect2.push_back(num_correct);
	}
	for (int k = 0; k < total_Kcorrect1.size(); k++)
	{
		cout << total_Kcorrect1[k] << "  ";
	}

}


long Loss(torch::Tensor class1, torch::Tensor class2, torch::Tensor test,int k)
{
	c10::IntArrayRef tsize = test.sizes();
	int num1 = 0;
	int num2 = 0;
	long correct_num = 0;
	auto distance = torch::tensor({ }).to(device_type);
	cout << distance.device() << endl;
	for (long i = 0; i < tsize[0]; i++)
	{
		auto distance1 = torch::sqrt(torch::square(class1.narrow(1, 0, 1) - test[i][0]) + torch::square(class1.narrow(1, 1, 1) - test[i][1])).to(device_type);
		auto distance2 = torch::sqrt(torch::square(class2.narrow(1, 0, 1) - test[i][0]) + torch::square(class2.narrow(1, 1, 1) - test[i][1])).to(device_type);
		auto distance = torch::cat({ distance1, distance2 }, 0);
		tuple<torch::Tensor, torch::Tensor> sort_ret = torch::sort(distance, 0, 0);    // {8000,1}

		//cout << distance << endl;
		
		for (int j = 0; j < k; j++)
		{
			int a = (get<1>(sort_ret)[j][0]).item<int>();
			if (a <= 4000)
			{
				num1 += 1;
			}
			else if (a > 4000)
			{
				num2 += 1;
			}
			if ( num1 > num2 && test[i][2].item<int>() == 1)
			{
				correct_num += 1;
			}
			else if (num1 < num2 && test[i][2].item<int>() == 2)
			{
				correct_num += 1;
			}
		}
	}
	return correct_num;
}
