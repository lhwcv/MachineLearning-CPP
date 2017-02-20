// lhwcv 2017-02-18
#include <Kmeans.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <time.h>
#include <assert.h>
#include <opencv2/opencv.hpp>
using  namespace std;
using  namespace cv;
namespace ml_cv{
	int DataSet::load_dataset_from_txt(const char *filename)
	{
		std::vector<std::string> lines;
		StrTool::load_lines_from_txt(filename, lines);
		for (unsigned int i = 0; i < lines.size(); i++)
		{
			vector<string> datastr = StrTool::split(lines[i], "\t");
			
			vector<float> onedata;
			for (int j = 0; j < datastr.size()-1; j++)
			{
				onedata.push_back(atof( datastr[j].c_str() ));
			}
			data_.push_back(onedata);
			labels_.push_back(atof(datastr[datastr.size() - 1].c_str()));
		}
		return 0;
	}
	void DataSet::min_max_each_dim(int dimInx, float &min, float &max)
	{
		float min_ = FLT_MAX, max_ = FLT_MIN;
		for (unsigned int i = 0; i < data_.size(); i++)
		{
			float v = data_[i][dimInx];
			if (v < min_)
				min_ = v;
			if (v > max_)
				max_ = v;
		}
		min = min_;
		max = max_;
	}
	int DataSet::write_to_txt(const char *filename)
	{
		ofstream ofs(filename);
		if (!ofs.is_open())
			return -1;
		for (int i = 0; i < data_.size(); i++)
		{
			for (int j = 0; j < data_[i].size(); j++)
				ofs << data_[i][j] << "\t";
			ofs << labels_[i] << endl;
		}
		return 0;
	}
	const std::vector<std::vector<float> >& DataSet::get_data()
	{
		return data_;
	}
	const std::vector<int>& DataSet::get_label()
	{
		return labels_;
	}
	void StrTool::load_lines_from_txt(const char *txtPath, std::vector<std::string> &linesVec)
	{
		std::ifstream txt(txtPath);
		if (!txt)
		{
			std::cout << "No such txt!" << std::endl;
			return;
		}
		while (!txt.eof())
		{
			std::string line;
			getline(txt, line);
			if (line.size() != 0)
			{
				linesVec.push_back(line);
			}
		}
	}
	std::vector<std::string> StrTool::split(std::string& str, const char* c)
	{
		char *cstr, *p;
		vector<string> res;
		cstr = new char[str.size() + 1];
		strcpy(cstr, str.c_str());
		p = strtok(cstr, c);
		while (p != NULL)
		{
			res.push_back(p);
			p = strtok(NULL, c);
		}
		return res;
	}
	int Kmeans::gen_random_centers(DataSet &dataset, int K, std::vector<std::vector<float> > &centers)
	{
		srand((unsigned)time(NULL));
		vector<float> min(K,0), max(K,0);
		for (int j = 0; j < dataset.get_data()[0].size(); j++)
			dataset.min_max_each_dim(j,min[j],max[j]);
		for (int i = 0; i < K; i++)
		{
			vector<float> center;
			for (int j = 0; j < dataset.get_data()[0].size(); j++)
			{
				float v = (float)(rand() % RAND_MAX) / RAND_MAX*(max[j] - min[j]) + min[j];
				center.push_back(v);
			}
			centers.push_back(center);
		}
		return 0;
	}
	float Kmeans::euclidean_distance(const std::vector<float> &vec1, const std::vector<float> &vec2)
	{
		assert(vec1.size() == vec2.size());
		float dis = 0;
		for (unsigned int i = 0; i < vec1.size(); i++)
		{
			dis += (vec1[i] - vec2[i])*(vec1[i] - vec2[i]);
		}
		return sqrt(dis);
	}
	int Kmeans::nearest_center_id( const std::vector<float> &vec, const std::vector<std::vector<float> > &centers)
	{
		float minDis=FLT_MAX;
		int id = -1;
		for (unsigned int i = 0; i < centers.size(); i++)
		{
			float dis = euclidean_distance(vec, centers[i]);
			if (minDis > dis )
			{
				minDis = dis;
				id = i;
			}
		}
		return id;
	}
	int Kmeans::calculate_labels(DataSet &dataset, int K, std::vector<std::vector<float> > &centers, std::vector<int> &labels)
	{
		for (unsigned int i = 0; i < dataset.get_data().size(); i++)
		{
			labels[i] =  nearest_center_id(dataset.get_data()[i], centers) ;
		}
		return 0;
	}
	float Kmeans::update_centers(DataSet &dataset, std::vector<int> &labels, std::vector<std::vector<float> > &centers)
	{
		vector<vector<float> >  averagePosition(centers.size(), vector<float>(centers[0].size(),0) );
		vector<int>  numOfEachCluster(centers.size(),0);

		for (unsigned int i = 0; i < dataset.get_data().size(); i++)
		{
			for (unsigned int j = 0; j < dataset.get_data()[0].size(); j++)
			{
				
				averagePosition[ labels[i] ][j] += dataset.get_data()[i][j];
				
			}
			numOfEachCluster[labels[i]] ++;
		}
		float centerMovement = 0;
		for (unsigned int i = 0; i < averagePosition.size(); i++)
		{
			// if some cluster appeared to be empty then:
			//   1. find the biggest cluster
			//   2. find the farthest from the center point in the biggest cluster
			//   3. exclude the farthest point from the biggest cluster and form a new 1-point cluster.
			if (numOfEachCluster[i] <= MIN_SAMPLES_IN_ONE_CLUSTER)
			{
				int maxNum = INT_MIN;
				int maxNumId = -1;
				for (unsigned int k = 0; k < numOfEachCluster.size(); k++)
				{
					if (maxNum < numOfEachCluster[k])
					{
						maxNum = numOfEachCluster[k];
						maxNumId = k;
					}
				}
				float maxDis = FLT_MIN;
				int maxDisId = -1;
				for (unsigned int m = 0; m < dataset.get_data().size(); m++)
				{
					if (labels[m] == maxNumId)
					{
						float dis = euclidean_distance(dataset.get_data()[m], centers[maxNumId]);
						if (dis > maxDis)
						{
							maxDis = dis;
							maxDisId = m;
						}
					}
				}
				for (unsigned int m = 0; m < centers[i].size(); m++)
				{
					centers[i][m] = dataset.get_data()[maxDisId][m];
				}
				//calculate_labels(dataset, centers.size(), centers, labels);
				//centerMovement += update_centers(dataset, labels, centers);
			}
			else
			{
		       for (unsigned int j = 0; j < averagePosition[0].size(); j++)
		       {
			        float v = averagePosition[i][j] / numOfEachCluster[i];
			        centerMovement += (centers[i][j] - v)*(centers[i][j] - v);
			        centers[i][j] = v;
		       }
	        }
		}
		
		return sqrt(centerMovement) / centers.size();
	}

	
	int Kmeans::run_kmeans(DataSet &dataset, int K, std::vector<std::vector<float> > & centers,
		std::vector<int> &labels,ShowKmeansFunc showfunc,float eps	)
	{
		assert(K>=2);
		if (K <2)
			return -1;
		gen_random_centers(dataset, K, centers);
		float changeValue = FLT_MAX;
		for (int i = 0; i < dataset.get_data().size(); i++)
			labels.push_back(0);
		while (changeValue > eps)
		{
			calculate_labels(dataset, K, centers, labels);
			if (showfunc != NULL)
				showfunc(dataset, centers, labels);
			changeValue = update_centers(dataset, labels, centers);
			cout <<"change Value:"<< changeValue << endl;
		}
		
		return 0;
	}
}; // namespace ml_cv;