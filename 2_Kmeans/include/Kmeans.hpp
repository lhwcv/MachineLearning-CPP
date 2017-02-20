// lhwcv 2017-02-18
#ifndef KMEANS_H
#define KMEANS_H
#include <iostream>
#include <vector>
#define RED   Scalar(0,0,255)
#define GREEN Scalar(0,255,0)
#define BLUE   Scalar(255,0,0)
#define YELLOW Scalar(0,255,255)
namespace ml_cv{
	class StrTool
	{
	public:
		static void load_lines_from_txt(const char *txtPath, std::vector<std::string> &imgPathVec);
		static std::vector<std::string> split(std::string& str, const char* c);
	};
	class DataSet
	{
	public:
		int   load_dataset_from_txt(const char *filename);
		int   write_to_txt(const char *filename);
		const std::vector<std::vector<float> >& get_data();
		const std::vector<int>& get_label();
		
		// min and max in dataset for the specific dimInx
		void min_max_each_dim(int dimInx,float &min,float &max);
	private:
		std::vector<std::vector<float> > data_;
		std::vector<int> labels_;
		
	};
	enum DistanceType { EUCLIDEAN, MINKOWSKI,CITYBLOCK };
	typedef void(*ShowKmeansFunc)(DataSet &dataset, std::vector<std::vector<float> >& centers, std::vector<int> &labels);
	class Kmeans
	{
	public:
		Kmeans()
		{
			MIN_SAMPLES_IN_ONE_CLUSTER = 5;
		}
		Kmeans(int minSamplesOneCluster)
		{
			MIN_SAMPLES_IN_ONE_CLUSTER = minSamplesOneCluster;
		}
		int run_kmeans(DataSet &dataset, int K, std::vector<std::vector<float> >& centers,
			std::vector<int> &labels, ShowKmeansFunc showfunc = NULL,
			           float eps = 0.1); // now only support EUCLIDEAN
	private:
		float euclidean_distance(const std::vector<float> &vec1, const std::vector<float> &vec2);

		// three step for kmeans
		int gen_random_centers(DataSet &dataset, int K, std::vector<std::vector<float> > &centers); 
		int calculate_labels(DataSet &dataset, int K, std::vector<std::vector<float> > &centers, std::vector<int> &labels);
		float update_centers(DataSet &dataset, std::vector<int> &labels, std::vector<std::vector<float> > &centers);

		int nearest_center_id(const std::vector<float> &vec, const std::vector<std::vector<float> > &centers);
		

		int MIN_SAMPLES_IN_ONE_CLUSTER;
	}; // class Kmeans
};// namespace ml_cv

#endif