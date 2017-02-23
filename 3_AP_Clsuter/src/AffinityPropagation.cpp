#include <AffinityPropagation.h>
#include <algorithm>
namespace ml_cv{
	
	
	Matrix2d AffinityPropagation::load_data(const char *filename)
	{
		Matrix2d data_label = Matrix2d::load_mat_from_txt(filename);
		data_ = Matrix2d(data_label.rows(), data_label.cols()-1 );
		for (int i = 0; i < data_label.rows();i++)
		{
			for (int j = 0; j < data_label.cols()-1; j++)
			{
				data_[i][j] = data_label[i][j];
			}
		}
		
		S_ = Matrix2d(data_label.rows(), data_label.rows(),0.0f);
		A_ = Matrix2d(data_label.rows(), data_label.rows(),0.0f);
		R_ = Matrix2d(data_label.rows(), data_label.rows(),0.0f);
		return data_;
	}
	void AffinityPropagation::run(Matrix2d  & centersDst, std::vector<int> &labelsDst, int iters)
	{
		calculateS();
		for (int i = 0; i < iters; i++)
		{
			calculateR();
			calculateA();
			
		}
		findCenter(centersDst);
		// calculate labels acccording euclidean diastance 
		for (int i = 0; i < data_.rows(); i++)
		{
			int label = 0;
			float minDis = FLT_MAX;
			for (int j = 0; j < centersDst.rows(); j++)
			{
				float dis = euclidean_distance(data_.row_ptr(i), centersDst.row_ptr(j), data_.cols(), 1,1);
				if (dis < minDis)
				{
					minDis = dis;
					label = j;
				}
			}
			labelsDst.push_back(label);
		}

		
	}
	
	void AffinityPropagation::calculateS()
	{
		vector<float>  vecS;
		for (int i = 0; i < S_.rows()-1; i++)
			for (int j = i+1; j < S_.cols(); j++)
			{
				S_[i][j] = -euclidean_distance(data_.row_ptr(i), data_.row_ptr(j), data_.cols(), 1, 1);
				S_[j][i] = S_[i][j];
				vecS.push_back(S_[i][j]);
			}
		// Set medium of S_ as preferences
		sort(vecS.begin(), vecS.end());
		int size = vecS.size();
		float mediumS=0;
		if (size % 2 == 0)
			mediumS = 0.5*(vecS[size / 2] + vecS[size / 2 + 1]);
		else
			mediumS = vecS[size / 2 + 1];
		for (int i = 0; i < S_.rows(); i++)
			S_[i][i] = mediumS;
	}
	void AffinityPropagation::calculateR()
	{
		for (int i = 0; i<R_.rows();i++)
			for (int k = 0; k < R_.cols(); k++)
			{
				float max = -FLT_MAX;
				for (int m = 0; m < k; m++)
				{
					float tmp = A_[i][m] + S_[i][m];
					if (A_[i][m] + S_[i][m] > max)
						max = A_[i][m] + S_[i][m];
				}
				for (int m = k + 1; m < R_.rows(); m++)
				{
					float tmp = A_[i][m] + S_[i][m];
					if (A_[i][m] + S_[i][m] > max)
						max = A_[i][m] + S_[i][m];
				}
				R_[i][k] = lamada_*R_[i][k]  + (1-lamada_)*(S_[i][k] - max);
				
			}
	}
	void AffinityPropagation::calculateA()
	{
		for (int i = 0; i < A_.rows(); i++)
			for (int k = 0; k < A_.cols(); k++)
			{
				if (i == k)
				{
					float accumu = 0;
					for (int m = 0; m < k; m++)
						accumu += max(0.0f, R_[m][k]);
					for (int m = k + 1; m < A_.rows();m++)
						accumu += max(0.0f, R_[m][k]);
					A_[i][k] = lamada_*A_[i][k] +(1-lamada_)*accumu;
				}
				else
				{
					float accumu = 0;
					for (int m = 0; m < min(i,k); m++)
						accumu += max(0.0f, R_[m][k]);
					for (int m = min(i, k) + 1; m < max(i, k); m++)
						accumu += max(0.0f, R_[m][k]);
					for (int m = max(i, k) + 1; m < A_.rows(); m++)
						accumu += max(0.0f, R_[m][k]);
					A_[i][k] = lamada_*A_[i][k] + (1 - lamada_)*min(0.0f, R_[k][k]+accumu);
				}
			}
	}
	void AffinityPropagation::findCenter(Matrix2d  & centersDst)
	{
		vector<vector<float> > centers;
		for (int i = 0; i < data_.rows(); i++)
		{
			float E = A_[i][i] + R_[i][i];
			//cout << E << endl;
			if (E > 0)
			{
				vector<float> center;
				for (int j = 0; j < data_.cols(); j++)
					center.push_back(data_[i][j]);
				centers.push_back(center);
			}
		}
		if (centers.size()>0)
		  centersDst = Matrix2d(centers);

	}


}; // namespace ml_cv