// lhwcv  2017-02-21
#ifndef MATRIX_H
#define MATRIX_H
#include <stdlib.h>
#include <iostream>
#include <memory>
#include <assert.h>
#include<iomanip>
#include<math.h>
#include <cstring>
#include <sstream>
#include <tools/StrTool.hpp>
namespace ml_cv
{

template<class Dtype>
class Matrix_
{
public:
    Matrix_()
    {
        rows_ = 0;
        cols_ = 0;
        counts_ = 0;
        data_ = nullptr;
    }
    inline Matrix_(int rows,int cols)
    {
        rows_ = rows;
        cols_ = cols;
        counts_ = rows*cols;
        data_  = new Dtype[counts_];
        memset((void*)data_,0,counts_*sizeof(Dtype));

    }
    inline Matrix_(int rows,int cols,const Dtype* data)
    {
        rows_ = rows;
        cols_ = cols;
        counts_ = rows*cols;
        data_  = new Dtype[counts_];
        memcpy(data_,data,counts_*sizeof(Dtype));

    }
	inline Matrix_(vector<vector<float> >  & dataVec)
	{
		assert(dataVec.size() > 0);
		rows_ = dataVec.size();
		cols_ = dataVec[0].size();
		counts_ = rows_*cols_;
		data_ = new Dtype[counts_];
		for (int i = 0; i < rows_; i++)
			for (int j = 0; j < cols_; j++)
				data_[i*cols_ + j] = dataVec[i][j];
	}
	
    inline Matrix_(int rows,int cols,Dtype value)
    {
        rows_ = rows;
        cols_ = cols;
        counts_ = rows*cols;
        data_  = new Dtype[counts_];
        for(int i=0;i<counts_;i++)
            data_[i] = value;
    }
    ~Matrix_()
    {
        rows_ = 0;
        cols_ = 0;
        counts_ = 0;

        if (data_ != nullptr)
        {
            delete[] data_;
            data_ = nullptr;
        }

    }
    Matrix_(const Matrix_& mat)
    {
        if(counts_!=mat.counts())
        {
            //cout<<"Warnning! Matrix size not match!"<<endl;
            delete [] data_;
            rows_ = mat.rows();
            cols_ = mat.cols();
            counts_ = mat.counts();
            data_ = new Dtype[counts_];
        }     
        memcpy(get_ptr(),mat.get_ptr(),mat.counts()*sizeof(Dtype));

    }
	static Matrix_ load_mat_from_txt(const char *filename)
	{
		vector<string> lines;
		StrTool::load_lines_from_txt(filename, lines);
		Matrix_ mat(lines.size(),StrTool::split(lines[0],"\t").size() );
		for (unsigned int i = 0; i < lines.size(); i++)
		{
			vector<string> dataStr = StrTool::split(lines[i], "\t");
			for (unsigned int j = 0; j < dataStr.size(); j++)
			{
				stringstream ss;
				ss << dataStr[j];
				Dtype d;
				ss >> d;
				mat[i][j] = d;
			}
		}
		return mat;
		
	}
	Dtype *row_ptr(int row)
	{
		return &data_[row*cols_];
	}
	Dtype *col_ptr(int col)
	{
		return &data_[col];
	}


    Dtype *get_ptr() const
    {
        return data_;
    }
    Matrix_ trans()
    {
        Matrix_<Dtype> transMat(this->cols_,this->rows_);
        for(int i=0;i<transMat.rows();i++)
            for(int j=0;j<transMat.cols();j++)
                transMat[i][j] = this->data_at(j,i);
        return transMat;

    }
    inline Dtype data_at(int row,int col) const
    {
        return data_[row*cols_+ col];
    }
    inline void set_data(Dtype v,int row,int col)
    {
        data_[row*cols()+col] = v;
    }

    inline Dtype *  operator[](int k) const
    {
        return &data_[k * cols_];
    }

    void gauss_random_init(Dtype E,Dtype V)
    {
        for(int i=0;i<counts_;i++)
            data_[i] = MathFunc::gaussrand(E,V);
    }
    void normal_random_init(Dtype a,Dtype b)
    {
        for(int i=0;i<counts_;i++)
            data_[i] = ((Dtype)(rand()%RAND_MAX) )/RAND_MAX*(b-a)+a; //
    }

    Matrix_ col_accumulate()
    {
        Matrix_ ret(this->row(),1);

        return ret;
    }

    template<typename T>
    friend std::ostream &operator<<(std::ostream &os,const Matrix_<T> &blob);

    template<typename T>
    friend Matrix_<T> operator *(const Matrix_<T> &mat1,const Matrix_<T> &mat2);

    template<typename T>
    friend Matrix_<T> operator *(const T v,const Matrix_<T> &mats);

    Matrix_ &operator =(const Matrix_& mat)
    {
        if(counts_!=mat.counts())
        {
            //cout<<"Warnning! Matrix size not match!"<<endl;
            this->~Matrix_();
            rows_ = mat.rows();
            cols_ = mat.cols();
            counts_ = mat.counts();
            data_ = new Dtype[counts_];
        }
        memcpy(get_ptr(),mat.get_ptr(),mat.counts()*sizeof(Dtype));
        return *this;

    }


    Matrix_ operator -(const Matrix_& mat)
    {
        assert(cols()==mat.cols()&&rows()==mat.rows());
        Matrix_ sub(mat.rows(),mat.cols());
        for(int i=0;i<sub.rows();i++)
            for(int j=0;j<sub.cols();j++)
                sub[i][j] = this->data_at(i,j)-mat[i][j];
        return sub;
    }
    Matrix_ operator +(const Matrix_& mat)
    {
        assert(cols()==mat.cols()&&rows()==mat.rows());
        Matrix_ add(mat.rows(),mat.cols());
        for(int i=0;i<add.rows();i++)
            for(int j=0;j<add.cols();j++)
                add[i][j] = this->data_at(i,j)+mat[i][j];
        return add;
    }

    Matrix_ operator -(void)
    {
        Matrix_ ret(rows(),cols());
        for(int i=0;i<counts();i++)
            ret.get_ptr()[i] = -(this->get_ptr()[i]);
        return ret;

    }
    Matrix_ operator -(int v)
    {
        Matrix_ sub(rows(),cols());
        for(int i=0;i<sub.rows();i++)
            for(int j=0;j<sub.cols();j++)
                sub[i][j] = this->data_at(i,j)-v;
        return sub;
    }

    Matrix_ operator -(const Dtype v)
    {
        Matrix_ sub(rows(),cols());
        for(int i=0;i<sub.rows();i++)
            for(int j=0;j<sub.cols();j++)
                sub[i][j] = this->data_at(i,j)-v;
        return sub;
    }
    Matrix_ operator /(const Dtype v)
    {
        return 1.0/v*(*this);
    }

//    template<typename  T>
//    friend Matrix_<T> operator -(const Dtype v,const Matrix_<T> &mat);

    Matrix_ operator &(const Matrix_ &mat)
    {
        return *this;
    }

    inline int rows() const
    {
        return rows_;
    }
    inline int cols() const
    {
        return cols_;
    }
    inline int counts() const
    {
        return counts_;
    }

    static Matrix_ multiply(const Matrix_ &A, const Matrix_ &B,Dtype scale)
    {

        assert(A.cols()==B.cols()&&A.rows()==B.rows());
        Matrix_ ret(A.rows(),A.cols());
        for(int i=0;i<ret.rows();i++)
            for(int j=0;j<ret.cols();j++)
            {
                ret[i][j] = A[i][j]*B[i][j]*scale;
            }
        return ret;
    }
    static Matrix_ accumulate_in_col(const Matrix_&A)
    {
        Matrix_ ret(A.rows(),1);
        for(int i=0;i<A.rows();i++)
        {
            Dtype ac = 0;
            for(int j=0;j<A.cols();j++)
                ac += A[i][j];
            ret[i][0] = ac;
        }
        return ret;

    }


    // C <- alpha* A ¡Á B + belta*C
    template<typename  T>
    static void gemm(const Matrix_<T>& A,const Matrix_<T> &B,Matrix_<T> &C,
                     Dtype alpha,Dtype belta)
    {
        if(C.rows()!=A.rows()||C.cols()!=B.cols()||!A.cols()==B.rows())
        {
            std::cout<<"Matrix size not match"<<std::endl;
            std::cout<<__FILE__<<" "<<__LINE__<<std::endl;
            assert(A.cols()==B.rows()&&C.rows()==A.rows()&&C.cols()==B.cols());
            return;
        }
        for(int row=0;row<C.rows();row++)
        {
            for(int col=0;col<C.cols();col++)
            {
                Dtype v=0;
                for(int i=0;i<A.cols();i++)
                    v += A[row][i]*B[i][col];
                C[row][col] = alpha*v+belta*C[row][col];
            }
        }

    }
    void add_col_vec(const Matrix_ &vec)
    {
        assert(vec.cols()==1&&vec.rows()==rows());
        for(int i=0;i<rows();i++)
            for(int j=0;j<cols();j++)
                set_data(vec[i][0]+data_at(i,j),i,j);
    }

private:


    Dtype* data_ = nullptr;
    int counts_;
    int rows_;
    int  cols_;

};

template<typename  T>
std::ostream &operator<<(std::ostream &os, const Matrix_<T> &mat)
{
    os<<"[ "<<std::endl;
    for(int i=0;i<mat.rows();i++)
    {
        for(int j=0;j<mat.cols();j++)
        {
            os<<std::setprecision(5)<<mat[i][j];
            if(j!=mat.cols()-1)
                os<<" ,";
        }
        os<<";"<<std::endl;
    };
    os<<"]"<<std::endl;
    return os;

}
template<typename T>
Matrix_<T> operator *(const Matrix_<T> &mat1,const Matrix_<T> &mat2)
{
    Matrix_<T> ret(mat1.rows(),mat2.cols());
    Matrix_<T>::gemm(mat1,mat2,ret,1,0);
    return ret;
}
template<typename T>
Matrix_<T> operator *(const T v,const Matrix_<T> &mat)
{
    Matrix_<T> ret(mat.rows(),mat.cols());
    for(int i=0;i<ret.rows();i++)
        for(int j=0;j<ret.cols();j++)
            ret[i][j] = v*mat[i][j];
    return ret;
}

typedef Matrix_<float>  Matrix2d;


};// namespace ml_cv

#endif // MATRIX_H
