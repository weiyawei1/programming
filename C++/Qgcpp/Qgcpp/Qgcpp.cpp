#include "G:\program\MATLAB\R2016a\extern\include\mex.h"
#include <stdlib.h>
#include<string.h>
#include<stdio.h>
#include <ctype.h>
#include <vector>
#include <algorithm>   
#include <minmax.h>
#include <stdlib.h>
#include <vector>

double Qg(double *inMatU,double *inMat,size_t mrowsU,size_t mrows,double *m,size_t m2,size_t n,size_t c);//计算qg值

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	//nlhs：输出参数个数
	//plhs：输出参数列表
	//nrhs：输入参数个数
	//prhs：输入参数列表   

	// Check for proper number of input and output arguments 
	if (nrhs != 6) {
		mexErrMsgIdAndTxt("MATLAB:mxcreatecellmatrix:nrhs",
			"输入参数个数不正确");
	}
	if (nlhs != 1) {
		mexErrMsgIdAndTxt("MATLAB:mxcreatecellmatrix:nlhs",
			"输出参数个数不正确");
	}

	//第一个输入变量的传递
	size_t mrowsU;    //行数
	size_t ncolsU;    //列数
	//mxArray *inMat;  //接收输入参数的指针
	double *inMatU;
	mrowsU = mxGetM(prhs[0]); //获取矩阵行数
	ncolsU = mxGetN(prhs[0]); //获取矩阵列数
	inMatU = mxGetPr(prhs[0]);//获取输入矩阵的指针

	//第二个输入变量的传递
	size_t mrows;    //行数
	size_t ncols;    //列数
	//mxArray *inMat;  //接收输入参数的指针
	double *inMat;
	mrows = mxGetM(prhs[1]); //获取矩阵行数
	ncols = mxGetN(prhs[1]); //获取矩阵列数
	inMat = mxGetPr(prhs[1]);//获取输入矩阵的指针
	/* 
	这里需要注意的是Matlab中矩阵的储存是列优先的，而C语言中是行优先的，在调用矩阵元素时需要注意：
	double result;
	 将iMat中的第 i行 j列的元素值赋给result 
	result = inMat[j*mrows+i]
	*/
	//第三个输入变量的传递
	double  *m = mxGetPr(prhs[2]);

	//第四个输入变量的传递
	double m2 = mxGetScalar(prhs[3]);
	//第五个输入变量的传递
	double n = mxGetScalar(prhs[4]);

	//第六个输入变量的传递
	double c = mxGetScalar(prhs[5]);

	double qg=0;
	//计算qg值
	qg = Qg(inMatU,inMat,mrowsU,mrows,m,(size_t)m2,(size_t)n,(size_t)c);
	
	double *p;
	plhs[0]=mxCreateDoubleMatrix(1,1,mxREAL);
	p=mxGetPr(plhs[0]);
	*p=qg;
}


/*
	用于计算Qg值 Qg = 每个社区的计算结果累加
	m:网络中n个节点的度
	m2: m2=2m=156
	mrows:邻接矩阵行数
	ncols:邻接矩阵列数
	*inMat:指向邻接矩阵第一个元素的地址
*/
double Qg(double *inMatU,double *inMat,size_t mrowsU,size_t mrows,double *m,size_t m2,size_t n,size_t c)
{
	double qgTemp=0;

	for(size_t k=0;k<c;c++){
		for(size_t i=0;i<n;i++){
			for(size_t j=0;j<n;j++){
				//Q = Q + (W(i,j) - (m(i).*m(j))./m2).*U(k,i).*U(k,j);
				qgTemp = qgTemp + (inMat[j*mrows+i] - (m[i-1] * m[j-1])/m2)*inMatU[i*mrowsU+k]*inMatU[j*mrowsU+k];
			}
		}
	}
	return qgTemp / m2;
}
