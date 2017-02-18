
#ifndef GENPOSITIONDATA_H
#define GENPOSITIONDATA_H
#include <vector>
#include <iostream>

#include <opencv2/opencv.hpp>

#define RED   Scalar(0,0,255)
#define GREEN Scalar(0,255,0)
#define BLUE   Scalar(255,0,0)
#define YELLOW Scalar(0,255,255)

namespace ml_cv{

	struct PositionData
	{
		float x, y;
		int classId;
	};
	std::vector<PositionData> get_position_data(const char *storeInFile);
	cv::Mat draw_position_data(std::vector<PositionData> &positionData);
	int write_position_data_to_txt(std::vector<PositionData> &positionData,const char *filepath);


};// namespace ml_cv

#endif