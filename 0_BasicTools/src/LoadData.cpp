#include <Tools.hpp>

using namespace cv;

namespace ml_cv{
/* 
 * load mnist data as vector<cv::Mat>  or vector<ml_cv::Matrix> or vector<vector<float> >
 */
int load_mnist(const char *filename,vector<cv::Mat> &data)
{
	return 0;
}
int load_mnist(const char *filename,vector<ml_cv::Matrix2d> &data)
{
	return 0;
}
/*
 * load txt data as flowing format
 * x1 x2 x3......xn  label
 */
int load_txt_data(const char *filename,ml_cv::Matrix2d &data,bool withLabel=true,char splitC='\t')
{
	return 0;
}

/* 
 * load iris data,It looks like:
 * 5.1,3.5,1.4,0.2,Iris-setosa
 */
int load_iris_data(Matrix2d &data,vector<string> &labels)
{
	return 0;
}
};// namespace ml_cv

