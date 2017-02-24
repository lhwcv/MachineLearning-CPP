// lhwcv 2017-02-22
#ifndef STRTOOL_H
#define STRTOOL_H
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
using namespace std;

namespace ml_cv{
	class StrTool
	{
	public:
		static void load_lines_from_txt(const char *txtPath, std::vector<std::string> &linesVec)
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
		static std::vector<std::string> split(const std::string& str, const char* c)
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


	};// class StrTool

};// namespace ml_cv

#endif