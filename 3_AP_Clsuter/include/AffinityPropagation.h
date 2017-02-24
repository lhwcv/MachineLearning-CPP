// lhwcv 2017-02-22
#ifndef AFFINITYPROPGATION_H
#define AFFINITYPROPGATION_H
#include <vector>
#include <Tools.hpp>
namespace ml_cv{
	template<typename Dtype>
	inline Dtype euclidean_distance(const Dtype *vec1, const Dtype *vec2, int len,int step1,int step2)
	{
		Dtype dis=0;
		for (int i = 0; i < len; i++)
		{
			//cout << vec1[step1*i] << endl;
			//cout << vec2[step2*i] << endl;
			dis += (vec1[step1*i] - vec2[step2*i])*(vec1[ step1*i] - vec2[ step2*i]);
		}
		return sqrt(dis);
	}

	Matrix2d load_mat_from_txt(const char *filename);
	class AffinityPropagation
	{
	public:
		AffinityPropagation()
		{
			lamada_ = 0.6;
		}
		Matrix2d load_data(const char *filename);
		void run(Matrix2d  & centersDst, std::vector<int> &labelsDst,int iters);
		inline void set_lamada(float lamada)
		{
			lamada_ = lamada;
		}
	private:
		float lamada_;
		Matrix2d data_;
		Matrix2d S_; // similarity
		Matrix2d R_; // responsibility
		Matrix2d A_; // availability
		void calculateS();
		void calculateR();
		void calculateA();
		void findCenter(Matrix2d  & centersDst);

	};// class AffinityPropagation


}; // namespace ml_cv

#endif