#include <iostream>
#include <opencv2/opencv.hpp>
#include <genPositionData.hpp>
using namespace std;
using namespace cv;
using namespace ml_cv;
int main()
{
	
	ml_cv::get_position_data("data.txt");
	return 0;
	
}