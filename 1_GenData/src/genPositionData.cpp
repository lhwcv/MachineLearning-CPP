#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <genPositionData.hpp>
using namespace cv;
using namespace std;
namespace ml_cv{

	int currentID=1;
	static const int  curtainSize = 640;
	void on_mouse(int event, int x, int y, int flags, void *userdata)
	{
		vector<PositionData> * positionDataP = (vector<PositionData> *)userdata;
		if (event == CV_EVENT_LBUTTONDOWN)
		{
			PositionData data;
			data.x = x; data.y = y;
			data.classId = currentID;
			positionDataP->push_back(data);
			Mat curtain = draw_position_data((*positionDataP));
			cv::imshow("curtain", curtain);
		}
	}
	std::vector<PositionData> get_position_data(const char *storeInFile)
	{
		std::vector<PositionData> positionData;
		Mat curtain(curtainSize, curtainSize, CV_8UC3);
		while (1)
		{
			namedWindow("curtain", 0);
			Mat curtain = draw_position_data(positionData);
			imshow("curtain", curtain);
			setMouseCallback("curtain", on_mouse, &positionData);
			char c = waitKey(0);
			currentID = (c - 48)%4;
			if (c==' ')
			{
				write_position_data_to_txt(positionData, storeInFile);
				break;
			}
		}
		return positionData;
	}
	int write_position_data_to_txt(std::vector<PositionData> &positionData, const char *filepath)
	{
		ofstream ofs(filepath);
		if (!ofs.is_open())
			return -1;
		ofs << "# x\t y\t label" << endl;
		for (unsigned int i = 0; i < positionData.size(); i++)
		{
			PositionData data = positionData[i];
			ofs << data.x << "\t" << data.y << "\t" << data.classId << endl;
		}
		return 0;
	}
	
	Mat draw_position_data(std::vector<PositionData> &positionData)
	{
		Mat curtain(curtainSize, curtainSize, CV_8UC3);
		for (unsigned int i = 0; i < positionData.size(); i++)
		{
			PositionData data = positionData[i];
			switch (data.classId)
			{
			case 1:
				circle(curtain, Point(data.x, data.y),3,RED,3);
				break;
			case 2:
				circle(curtain, Point(data.x, data.y), 3, BLUE, 3);
				break;
			case 3:
				circle(curtain, Point(data.x, data.y), 3, GREEN, 3);
				break;
			default:
				circle(curtain, Point(data.x, data.y), 3, YELLOW, 3);
				
			}
		}
		return curtain;
	}
	

};// namespace ml_cv



