// lhwcv 2017-02-23
#include <iostream>
#include <vector>
#include<opencv2/opencv.hpp>
#include <Matrix.hpp>
#include <AffinityPropagation.h>
#define RED   Scalar(0,0,255)
#define GREEN Scalar(0,255,0)
#define BLUE   Scalar(255,0,0)
#define YELLOW Scalar(0,255,255)

using namespace std;
using namespace ml_cv;
using namespace cv;
void show_func(Matrix2d &dataset, Matrix2d & centers, std::vector<int> &labels);
void test_ap()
{
	AffinityPropagation ap;
	Matrix2d data = ap.load_data("data2.txt");
	Matrix2d   centers;
	std::vector<int> labels;
	ap.run(centers, labels, 20);
	show_func(data, centers, labels);
	getchar();
}
int main()
{
	test_ap();

}


void show_func(Matrix2d &dataset, Matrix2d & centers, std::vector<int> &labels)
{
	
	assert(dataset.rows() == labels.size());
	vector<Scalar> colors;
	colors.push_back(RED);
	colors.push_back(BLUE);
	colors.push_back(GREEN);
	colors.push_back(YELLOW);
	colors.push_back(Scalar(255,222,173) );
	colors.push_back(Scalar(47, 79, 79));
	colors.push_back(Scalar(0, 191, 255));
	colors.push_back(Scalar(0, 100, 0));

	colors.push_back(Scalar(255, 193, 37));
	colors.push_back(Scalar(205, 92, 92));
	colors.push_back(Scalar(255, 211,155));
	

	cv::Mat curtain(640, 640, CV_8UC3);
	for (unsigned int i = 0; i < dataset.rows(); i++)
	{
		Point p(dataset[i][0], dataset[i][1]);
		circle(curtain, Point(p.x, p.y), 3, colors[labels[i]%colors.size()], 3);
	
	}
	for (int i = 0; i < centers.rows(); i++)
		rectangle(curtain, Rect(centers[i][0] - 10, centers[i][1] - 10, 20, 20), colors[labels[i] % colors.size()], 3);

	namedWindow("draw", 0);
	imshow("draw", curtain);
	waitKey(0);
}







