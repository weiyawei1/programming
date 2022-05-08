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
double Q(size_t m,size_t n,size_t mrows,size_t ncols,double *inMat);//����qֵ
double wSum(size_t mrows,size_t m, size_t n,double *ptr); //�����ڽӾ���ĵ�m�е�n��Ԫ��֮��

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	//nlhs�������������
	//plhs����������б�
	//nrhs�������������
	//prhs����������б�   

	// Check for proper number of input and output arguments 
	if (nrhs != 3) {
		mexErrMsgIdAndTxt("MATLAB:mxcreatecellmatrix:nrhs",
			"���������������ȷ");
	}
	if (nlhs != 1) {
		mexErrMsgIdAndTxt("MATLAB:mxcreatecellmatrix:nlhs",
			"���������������ȷ");
	}

	//��һ����������Ĵ���
	size_t mrows;    //����
	size_t ncols;    //����
	//mxArray *inMat;  //�������������ָ��
	double *inMat;
	mrows = mxGetM(prhs[0]); //��ȡ��������
	ncols = mxGetN(prhs[0]); //��ȡ��������
	inMat = mxGetPr(prhs[0]);//��ȡ��������ָ��

	//������Ҫע�����Matlab�о���Ĵ����������ȵģ���C�������������ȵģ��ڵ��þ���Ԫ��ʱ��Ҫע�⣺
	//double result;
	// ��iMat�еĵ� i�� j�е�Ԫ��ֵ����result 
	//result = inMat[j*mrows+i]

	//�ڶ�����������Ĵ��ݣ����������
	double m = mxGetScalar(prhs[1]);

	//��������������Ĵ���(����)
	double n = mxGetScalar(prhs[2]);

	double q=0;
	q = Q((size_t)m,(size_t)n,mrows,ncols,inMat);
	
	double *p;
	plhs[0]=mxCreateDoubleMatrix(1,1,mxREAL);
	p=mxGetPr(plhs[0]);
	*p=q;
}


/*
	���ڼ���Qֵ
	m:�ڽӾ������
	n:���������
	mrows:�ڽӾ�������
	ncols:�ڽӾ�������
	*inMat:ָ���ڽӾ����һ��Ԫ�صĵ�ַ
*/
double Q(size_t m,size_t n,size_t mrows,size_t ncols,double *inMat)
{
	double qTemp=0.0000;
	for(size_t i=0;i<n;i++){
		for(size_t j=0;j<n;j++){
			//sum(adj(i,:)) //�����i��֮��
			//Q = Q + (adj(i,j) - sum(adj(i,:))*sum(adj(j,:))/(2*m))/(2*m);  
			qTemp = qTemp+(inMat[j*mrows+i]- wSum(mrows,i,ncols,inMat)*wSum(mrows,j,ncols,inMat)/(m<<1))/(m<<1);
		}
	}
	return qTemp;
}

//�����ڽӾ���ĵ�m�е�n��Ԫ��֮��
double wSum(size_t mrows,size_t m,size_t n, double *ptr)
{
	double sum=0;
	for(size_t i=0;i<n;i++){
		sum = sum + ptr[i*mrows+m-1];
	}
	return sum;
}

