// lhwcv 2017-02-18
#include <iostream>
#include <Kmeans.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace ml_cv;
using namespace std;

void show_func(DataSet &dataset, std::vector<std::vector<float> >& centers, std::vector<int> &labels);

int main()
{
	DataSet dataset;
	dataset.load_dataset_from_txt("data.txt");
	Kmeans app;
	std::vector<std::vector<float> > centers;
	std::vector<int> labels;
	int K = 4;
	app.run_kmeans(dataset, K,centers, labels,show_func);
	getchar();

	return 0;
}
// This show func only for less than 4 cluster
void show_func(DataSet &dataset, std::vector<std::vector<float> >& centers, std::vector<int> &labels)
{
	if (centers.size() > 4)
		return;
	vector<Scalar> colors;
	colors.push_back(RED);
	colors.push_back(BLUE);
	colors.push_back(GREEN);
	colors.push_back(YELLOW);

	cv::Mat curtain(640, 640, CV_8UC3);
	for (unsigned int i = 0; i < dataset.get_data().size(); i++)
	{
		Point p(dataset.get_data()[i][0], dataset.get_data()[i][1]);
		switch (labels[i])
		{
		case 0:
			circle(curtain, Point(p.x, p.y), 3, RED, 3);
			break;
		case 1:
			circle(curtain, Point(p.x, p.y), 3, BLUE, 3);
			break;
		case 2:
			circle(curtain, Point(p.x, p.y), 3, GREEN, 3);
			break;
		default:
			circle(curtain, Point(p.x, p.y), 3, YELLOW, 3);

		}
	}
	for (int i = 0; i < centers.size();i++)
		rectangle(curtain, Rect(centers[i][0] - 10, centers[i][1] - 10, 20, 20), colors[i], 3);

	namedWindow("draw", 0);
	imshow("draw", curtain);
	waitKey(0);
}
