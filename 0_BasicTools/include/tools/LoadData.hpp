#ifndef LOADDATA_H
#define LOADDATA_H
#include <vector>
#include <string>
using namespace std;
#include <tools/Matrix.hpp>
#include <tools/StrTool.hpp>
namespace ml_cv{
/* 
 * load mnist data as vector<cv::Mat>  or vector<ml_cv::Matrix> or vector<vector<float> >
 */
int load_mnist(const char *filename,vector<cv::Mat> &data);
int load_mnist(const char *filename,vector<ml_cv::Matrix> &data);
/*
 * load txt data as flowing format
 * x1 x2 x3......xn  label
 */
int load_txt_data(const char *filename,ml_cv::Matrix &data,bool withLabel=true,char splitC='\t');

/* 
 * load iris data,It looks like:
 * 5.1,3.5,1.4,0.2,Iris-setosa
 */
int load_iris_data(Matrix &data,vector<string> &labels);
};// namespace ml_cv
#endif

