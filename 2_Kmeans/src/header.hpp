/*
* lhwcv 2017-02-2
* This is a tool used to generate .hpp .cpp files!
*/
#include <vector>
#include <Matrix.hpp>

namespace ml_cv
{
	class AffinityPropagation
	{
		public:
		     void load_data(const char*filename);
             void run();
		private:
		     Matrix2d S_;
		
	};//class AffinityPropagation
	
	int loop_print()
	{
		for(int i=0;i<10;i++)
		{
			for(int j=0;j<10;j++)
			{
				std::cout<<"hello"<<endl;
			}
		}
		for(int i=0;i<10;i++)
		{
			for(int j=0;j<10;j++)
			{
				std::cout<<"hello"<<endl;
			}
		}
	}
	
};//namespace ml_cv
