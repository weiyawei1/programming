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
double Q(size_t m,size_t n,size_t mrows,size_t ncols,double *inMat);//计算q值
double wSum(size_t mrows,size_t m, size_t n,double *ptr); //计算邻接矩阵的第m行的n个元素之和

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	//nlhs：输出参数个数
	//plhs：输出参数列表
	//nrhs：输入参数个数
	//prhs：输入参数列表   

	// Check for proper number of input and output arguments 
	if (nrhs != 3) {
		mexErrMsgIdAndTxt("MATLAB:mxcreatecellmatrix:nrhs",
			"输入参数个数不正确");
	}
	if (nlhs != 1) {
		mexErrMsgIdAndTxt("MATLAB:mxcreatecellmatrix:nlhs",
			"输出参数个数不正确");
	}

	//第一个输入变量的传递
	size_t mrows;    //行数
	size_t ncols;    //列数
	//mxArray *inMat;  //接收输入参数的指针
	double *inMat;
	mrows = mxGetM(prhs[0]); //获取矩阵行数
	ncols = mxGetN(prhs[0]); //获取矩阵列数
	inMat = mxGetPr(prhs[0]);//获取输入矩阵的指针

	//这里需要注意的是Matlab中矩阵的储存是列优先的，而C语言中是行优先的，在调用矩阵元素时需要注意：
	//double result;
	// 将iMat中的第 i行 j列的元素值赋给result 
	//result = inMat[j*mrows+i]

	//第二个输入变量的传递（网络边数）
	double m = mxGetScalar(prhs[1]);

	//第三个输入变量的传递(代数)
	double n = mxGetScalar(prhs[2]);

	double q=0;
	q = Q((size_t)m,(size_t)n,mrows,ncols,inMat);
	
	double *p;
	plhs[0]=mxCreateDoubleMatrix(1,1,mxREAL);
	p=mxGetPr(plhs[0]);
	*p=q;
}


/*
	用于计算Q值
	m:邻接矩阵边数
	n:脑网络代数
	mrows:邻接矩阵行数
	ncols:邻接矩阵列数
	*inMat:指向邻接矩阵第一个元素的地址
*/
double Q(size_t m,size_t n,size_t mrows,size_t ncols,double *inMat)
{
	double qTemp=0.0000;
	for(size_t i=0;i<n;i++){
		for(size_t j=0;j<n;j++){
			//sum(adj(i,:)) //计算第i行之和
			//Q = Q + (adj(i,j) - sum(adj(i,:))*sum(adj(j,:))/(2*m))/(2*m);  
			qTemp = qTemp+(inMat[j*mrows+i]- wSum(mrows,i,ncols,inMat)*wSum(mrows,j,ncols,inMat)/(m<<1))/(m<<1);
		}
	}
	return qTemp;
}

//计算邻接矩阵的第m行的n个元素之和
double wSum(size_t mrows,size_t m,size_t n, double *ptr)
{
	double sum=0;
	for(size_t i=0;i<n;i++){
		sum = sum + ptr[i*mrows+m-1];
	}
	return sum;
}

