#include <bits/stdc++.h>
using namespace std;
/*
1	天况  晴  多云  雨   1 2 3
2	温度  热 中  冷      1 2 3
3	湿度  大 正常        1 2
4	风况  无 有          1 2
	分类  N   Y          0 1
*/
#define PARAGRAMS 4		   // 天况 温度 湿度 风况 四个参数
#define PARA_CHILD 3	   // 上面四个参数里面的可能情况里的最大值，天况和温度种数是一样大的，即3
#define TRAINNING_ITEMS 14 // 训练集条目，14项
#define TESTING_ITEMS 14   // 测试集条目，14项，这里直接把训练集用于测试
#define PREDICTING_ITEMS 6 // 需要预测的项目，6条

vector<vector<int>> predata();
// 保存的决策树文件
ofstream output_rules("output-Rules.txt");
// 保存的预测文件
ofstream output_pre("output-pre.txt");

struct node *createnode();
void print(vector<vector<int>> &v);
void put_to_vector();
void inputData();
vector<vector<int>> newDataSet2(vector<vector<int>> v, int attrnum, int attrval);
vector<vector<int>> newDataSet(vector<vector<int>> &v, int attrnum, int attrval);
float calcEntropy(vector<vector<int>> &v);
float calcGain(vector<vector<int>> &v, int attrnum);
int maxGain(vector<vector<int>> v);
void id3(vector<vector<int>> the_set, struct node *temp, vector<int> print, int n_t);
float calculate_accuracy(vector<vector<int>> vec, struct node *root);
vector<vector<int>> testdata();
void predict(vector<vector<int>> vec, struct node *root);
vector<vector<int>> predata();

/*
这里的下标是在attrs_new对应；

下标	属性		  分别对应值
1		天况
				晴  多云  雨   1 2 3
2		温度
				热 中  冷      1 2 3
3		湿度
				大 正常        1 2
4		风况
				无 有          1 2
0		分类
				N   Y          0 1
*/

// 以下全局变量一旦初始化 则不会有更改
vector<set<int>> attrs;

// 由上面定义的attrs派生的向量的向量。
// 在生成所有属性后，它与attrs具有相同的内容。
// 与遍历集合相比，这种实现更容易遍历向量。
vector<vector<int>> attrs_new;

// 训练集
vector<vector<int>> dataset;

// 树节点
struct node
{
	/*
	如果它是一个叶节点，我们需要检查它是一个真或假或NULL节点。
	查找它是内部节点还是外部节点
	以惟一地标识属性。
	保存该项的指针。
	*/
	int id; // 如果id为1，对应的就是天况节点，print向量中有分支描述

	// 初始为NULL，节点树子节点的指针数组，PARA_CHILD即为所有节点中所有可能种数的最大值
	struct node *array[PARA_CHILD];
};

struct node *createnode()
{
	struct node *temp = (struct node *)malloc(sizeof(struct node));
	for (int i = 0; i < PARA_CHILD; i++)
	{
		temp->array[i] = NULL;
	}
	return temp;
}

// 打印二维vector，debug用
// void print(vector<vector<int>> &v)
// {
// 	for (int i = 0; i < v.size(); ++i)
// 	{
// 		cout << i << ": ";
// 		for (int j = 0; j < v[i].size(); ++j)
// 		{
// 			cout << v[i][j] << " ";
// 		}
// 		cout << "\n";
// 	}
// }

// 集合的向量转换为向量的向量，以后使用attrs_new
void put_to_vector() // 仅用于 attrs -> attrs_new
{
	vector<int> attrs_temp;
	for (int i = 0; i < attrs.size(); i++)
	{
		attrs_temp.clear();
		std::copy(attrs[i].begin(), attrs[i].end(), std::back_inserter(attrs_temp));
		attrs_new.push_back(attrs_temp);
	}
}

// 接受训练数据的输入，并将值存储在“dataset”中。初始化attrs向量。
void inputData()
{

	// 							 天况 温度 湿度 风况 分类
	// 单行训练项数据   data_01:	1	1	1	1	0
	vector<int> datasigleline;

	ifstream input("trainingdata.txt");
	int temp;
	set<int> values;
	for (int i = 0; i < PARAGRAMS + 1; ++i)
	{
		attrs.push_back(values);
	}
	for (int i = 0; i < TRAINNING_ITEMS; ++i)
	{
		datasigleline.clear();
		string cl;
		input >> cl; // 清除第一列，data_01:
		for (int j = 1; j < PARAGRAMS + 1; ++j)
		{
			// 后面五列加进去
			input >> temp;
			datasigleline.push_back(temp);
			attrs[j].insert(temp);
		}
		std::vector<int>::iterator it;
		it = datasigleline.begin();
		input >> temp;					// 分类结果
		datasigleline.insert(it, temp); // 下标为0 的保存分类结果
		attrs[0].insert(temp);			// 下标为0 的保存分类结果  类似于属性键值对

		dataset.push_back(datasigleline);
	}
	put_to_vector();
}

// 递归生成子节点时的必要操作，接受一个数据集，并返回一个新的数据集
// 比如说下标为5的因素是占当前节点最大的权重（这个权重的下标由maxGain计算得到），
//  那么就在属性键值对那 找到下标为5的因素对应啥具体元素，在id3函数中调用此函数，
//  第一个参数是当前节点对应的数据集，
//  第二个参数是下标3
//  第三个参数是下标为3的因素对应啥具体元素

// 返回的是如果当前节点对应的数据集中的某行中的下标3的元素 是 attrval，
// 那就合成所有这样的行为一个二维vector并返回，全局变量ds2做的就是这件事
vector<vector<int>> newDataSet2(vector<vector<int>> v, int attrnum, int attrval)
{
	vector<vector<int>> ds2;
	ds2.clear();
	for (int i = 0; i < v.size(); ++i)
	{
		if (v[i][attrnum] == attrval)
		{
			ds2.push_back(v[i]);
		}
	}
	return ds2;
}

// 返回与特定属性的给定attrval匹配的行数
// 例如 v是数据集 attrnum 是天况  	attrval 是 天况可能的类型 晴
vector<vector<int>> newDataSet(vector<vector<int>> &v, int attrnum, int attrval)
{
	vector<vector<int>> dstemp;
	for (int i = 0; i < v.size(); ++i)
	{
		if (v[i][attrnum] == attrval)
		{
			dstemp.push_back(v[i]); // 保存对应的训练项
		}
	}
	return dstemp;
}

// 计算一个特定集合的熵
// Entropy(S) =  - ∑ (p(i) * log2 p(i))    i~[0-1],两种分类情况
float calcEntropy(vector<vector<int>> &v)
{

	float e = 0;
	// 在v初始化的时候，v[0]表示分类
	// newDataSet(v, 0, 0),其实是对应于newDataSet(v, 分类 , 0)
	float a0 = newDataSet(v, 0, 0).size(); // 后续计算(p(0) * log2 p(0))中的p0的分子 ， 表示v中分类为0的项数

	// newDataSet(v, 0, 1),其实是对应于newDataSet(v, 分类 , 1)
	float a1 = newDataSet(v, 0, 1).size(); // 后续计算(p(1) * log2 p(1))中的p1的分子 ， 表示v中分类为1的项数

	//一般情况下 a0 + a1 == v.size()
	if (a0 == 0 || a1 == 0)
		return 0;
	e = -(((a0 / v.size()) * (log(a0 / v.size()) / log(2))) + ((a1 / v.size()) * (log(a1 / v.size()) / log(2))));
	return e; // Entropy(S) =  - ∑ (p(i) * log2 p(i))
}



// 返回数据集和参数中传递的所需属性计算的增益
// Gain(S, A) = Entropy(S) - ∑((|Sv| / |S|) * Entropy(Sv)
float calcGain(vector<vector<int>> &v, int attrnum)
{
	float g;
	float calc = 0;
	vector<vector<int>> ds1;
	for (int i = 1; i <= attrs_new[attrnum].size(); ++i)
	{
		// ds1中包含了传过来的训练集中v的所有与attrnum相关的i的属性的训练项
		/*
		例如：
			newDataSet(v, 1 , 1)
			即为
			newDataSet(v, 天况 , 1)

			下面是传过来的v
			分类 天况 温度 湿度 风况
			0	1	1	1	1
			0	1	1	1	2
			1	2	1	1	1

			那么ds1就是只包含下面两项，因为天况对应的第三项是2，不是1
			0	1	1	1	1
			0	1	1	1	2
		*/
		ds1 = newDataSet(v, attrnum, i); // Sv
		float gaina = ds1.size();		 //   |Sv|
		float ab = calcEntropy(ds1);	 //  Entropy(Sv)
		calc += (gaina / v.size()) * ab; // ∑((|Sv| / |S|) * Entropy(Sv)
	}
	g = calcEntropy(v) - calc; // Gain(S, A) = Entropy(S) - ∑((|Sv| / |S|) * Entropy(Sv)
	return g;
}

// 函数是一个模块化组件，它告诉需要在树中选择的下一个属性。
// 具有最高增益的 是在递归ID3调用中 被选择和使用的那个。
// 它计算所有属性的增益，并返回相应集合的属性下标。
int maxGain(vector<vector<int>> v)
{
	float gains[PARAGRAMS + 1];
	gains[0] = 0;
	if (v.size() == 0)
	{
		return -3;
	}
	for (int i = 1; i < PARAGRAMS + 1; ++i)
	{
		gains[i] = calcGain(v, i); // 对四个参数分别计算增益
	}

	int flag = 1;
	float maxgains;
	maxgains = gains[1];

	for (int i = 1; i < PARAGRAMS + 1; ++i)
	{
		if (gains[i] > maxgains)
		{
			maxgains = gains[i];
			flag = i;
		}
	}

	if (maxgains == 0)
	{
		if (v.size() == 0)
		{
			return -3; // 出错了
		}
		if (v[0][0] == 0)
		{
			return -2; // 分到 N
		}
		if (v[0][0] == 1)
		{
			return -1; // 分到 Y
		}
	}
	return flag; // 返回影响因子最大的元素，如天况等
}

/*
	递归ID3函数，它使用上面讨论的所有组件来生成决策树。每个节点都被设计为一个属性，并且它具有它可以接受的所有属性值。
	它接收以下输入。
	1)一个可能被简化的数据集。
	2)节点指针。
	3)矢量打印，帮助我们打印规则。
	4)一个n_t变量，帮助我们跟踪构建树所执行的DFS调用,在print向量中，存储必要信息。
*/
void id3(vector<vector<int>> the_set, struct node *temp, vector<int> print, int n_t)
{

	int id_max = maxGain(the_set); // Id_max 特定集合(the_set)的最大增益的属性。
	struct node *mytemp = NULL;
	if (n_t == 0)
	{
		print.push_back(id_max);
	}
	else
	{
		print.push_back(n_t);
		print.push_back(id_max);
	}

	temp->id = id_max;
	int n = 0;
	if (id_max > 0)
	{
		n = attrs_new[id_max].size();
	}
	if (id_max == -1)
	{
		// 分类到第一类，也就是 Y
		for (int j = 0; j < print.size() - 1; j++)
		{
			if (j == 0)
				output_rules << "IF NODE_ID" << print[j] << " = "; // output_rules 表示输出决策规则 文件流
			else if (j % 2 == 0)
				output_rules << "  and NODE_ID" << print[j] << " = ";
			else
				output_rules << print[j];
		}

		output_rules << "\t\t THEN => 1 (Y) is target class" << endl;
	}
	else if (id_max == -2)
	{
		// 分类到第2类，也就是  N
		for (int j = 0; j < print.size() - 1; j++)
		{
			if (j == 0)
				output_rules << "IF NODE_ID" << print[j] << " = ";
			else if (j % 2 == 0)
				output_rules << "  and NODE_ID" << print[j] << " = ";
			else
				output_rules << print[j];
		}

		output_rules << "\t\t THEN => 0 (N) is target class" << endl;
	}
	//决策树生成
	for (int i = 0; i < n; i++)
	{
		(temp->array)[i] = createnode();
		mytemp = (temp->array)[i];
		/*
		创建新的集合的函数。
		第一个是原始集
		第二个是需要使用的属性。
		第三个是需要使用的属性的值。

		这里调用NEWDataset2返回一个压缩的数据集，剥离了我们刚刚选择的原始属性 => id_max;
		*/
		id3(newDataSet2(the_set, id_max, attrs_new[id_max][i]), mytemp, print, i + 1);
	}
}

/*
	函数接受测试集的输入，以及指向我们已经生成的决策树的指针。
	遍历树，直到到达一个标记为-1或-2的叶节点。
	一旦它到达一个叶子，它就会检查预测的目标值是否与从树中导出的目标值相同。
	如果是这样，它会加到计数器上。
	最终返回 accuracy = (计数器 / 测试项数);
*/
float calculate_accuracy(vector<vector<int>> vec, struct node *root)
{
	float accuracy;
	float denom = vec.size();
	float counter = 0;
	struct node *temp = NULL;
	for (int i = 0; i < vec.size(); i++)
	{

		temp = root;
		while ((temp->id) > 0) //决策节点规划路径
		{
			int i_d = temp->id - 1;
			if (i_d < 0)
			{
				break;
			}
			int ii = vec[i][i_d];
			ii--; // 取得下标
			temp = (temp->array)[ii];
		}

		if ((temp->id == -1 && vec[i][PARAGRAMS] == 1) || (temp->id == -2 && vec[i][PARAGRAMS] == 0))
		{
			counter++;
		}
	}
	accuracy = (counter / denom);
	return accuracy;
}

/*
	从testingdata.txt中读取testdata。
	输入流被创建并从那里读取输入。
	它返回一个向量的向量，其中包含测试数据的所有行。
*/
vector<vector<int>> testdata()
{
	ifstream inputtest("testingdata.txt");
	vector<vector<int>> full_test;
	vector<int> test_container;
	for (int i = 0; i < TESTING_ITEMS; i++)
	{
		test_container.clear();
		string s;
		inputtest >> s;
		for (int i = 0; i < PARAGRAMS + 1; i++)
		{
			int temp;
			inputtest >> temp;
			test_container.push_back(temp);
		}
		full_test.push_back(test_container);
	}
	return full_test;
}

// 预测，类似测试
void predict(vector<vector<int>> vec, struct node *root)
{
	struct node *temp = NULL;
	for (int i = 0; i < vec.size(); i++)
	{

		temp = root;
		while ((temp->id) > 0)
		{
			int i_d = temp->id - 1;
			if (i_d < 0)
			{
				break;
			}
			int ii = vec[i][i_d];
			ii--; // 取得下标
			temp = (temp->array)[ii];
		}

		if (temp->id == -1)
			vec[i][PARAGRAMS] = 1;
		else if (temp->id == -2)
			vec[i][PARAGRAMS] = 0;
	}

	for (int i = 0; i < vec.size(); i++)
	{
		output_pre << "data_0" << i + 1 << ":\t";
		for (int j = 0; j < vec[0].size(); j++)
		{
			if (j == vec[0].size() - 1)
			{
				if (vec[i][j] == 1)
					output_pre << "Y";
				else if (vec[i][j] == 0)
					output_pre << "N";
			}
			else
				output_pre << vec[i][j] << "\t";
		}
		output_pre << endl;
	}
}

// 与 测试集的文件输入到二维向量中类似，但是predictingdata.txt中的分类结果默认为N，
//  也就是最右边的列全0，后期调用predict函数预测时，写入到output-pre.txt文件中
vector<vector<int>> predata()
{
	ifstream inputpre("predictingdata.txt");
	vector<vector<int>> full_pre;
	vector<int> pre_container;
	for (int i = 0; i < PREDICTING_ITEMS; i++)
	{
		pre_container.clear();
		string s;
		inputpre >> s;
		for (int i = 0; i < PARAGRAMS + 1; i++)
		{
			int temp;
			inputpre >> temp;
			pre_container.push_back(temp);
		}
		full_pre.push_back(pre_container);
	}
	return (full_pre);
}

// 所有的文件流 没有必要关闭，问题不大。
int main()
{

	ofstream outputacc("accuracy.txt");
	inputData();

	struct node *root = (struct node *)malloc(sizeof(struct node));
	for (int i = 0; i < PARA_CHILD; i++)
	{
		root->array[i] = NULL;
	}

	vector<int> print;

	id3(dataset, root, print, 0);

	cout << endl;

	float acc = calculate_accuracy(testdata(), root);
	outputacc << "测试集测试的准确度为：" << acc * 100 << "%";

	predict(predata(), root);
	return 0;
}
