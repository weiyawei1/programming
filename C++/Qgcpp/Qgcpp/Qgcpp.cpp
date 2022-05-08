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

double Qg(double *inMatU,double *inMat,size_t mrowsU,size_t mrows,double *m,size_t m2,size_t n,size_t c);//����qgֵ

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	//nlhs�������������
	//plhs����������б�
	//nrhs�������������
	//prhs����������б�   

	// Check for proper number of input and output arguments 
	if (nrhs != 6) {
		mexErrMsgIdAndTxt("MATLAB:mxcreatecellmatrix:nrhs",
			"���������������ȷ");
	}
	if (nlhs != 1) {
		mexErrMsgIdAndTxt("MATLAB:mxcreatecellmatrix:nlhs",
			"���������������ȷ");
	}

	//��һ����������Ĵ���
	size_t mrowsU;    //����
	size_t ncolsU;    //����
	//mxArray *inMat;  //�������������ָ��
	double *inMatU;
	mrowsU = mxGetM(prhs[0]); //��ȡ��������
	ncolsU = mxGetN(prhs[0]); //��ȡ��������
	inMatU = mxGetPr(prhs[0]);//��ȡ��������ָ��

	//�ڶ�����������Ĵ���
	size_t mrows;    //����
	size_t ncols;    //����
	//mxArray *inMat;  //�������������ָ��
	double *inMat;
	mrows = mxGetM(prhs[1]); //��ȡ��������
	ncols = mxGetN(prhs[1]); //��ȡ��������
	inMat = mxGetPr(prhs[1]);//��ȡ��������ָ��
	/* 
	������Ҫע�����Matlab�о���Ĵ����������ȵģ���C�������������ȵģ��ڵ��þ���Ԫ��ʱ��Ҫע�⣺
	double result;
	 ��iMat�еĵ� i�� j�е�Ԫ��ֵ����result 
	result = inMat[j*mrows+i]
	*/
	//��������������Ĵ���
	double  *m = mxGetPr(prhs[2]);

	//���ĸ���������Ĵ���
	double m2 = mxGetScalar(prhs[3]);
	//�������������Ĵ���
	double n = mxGetScalar(prhs[4]);

	//��������������Ĵ���
	double c = mxGetScalar(prhs[5]);

	double qg=0;
	//����qgֵ
	qg = Qg(inMatU,inMat,mrowsU,mrows,m,(size_t)m2,(size_t)n,(size_t)c);
	
	double *p;
	plhs[0]=mxCreateDoubleMatrix(1,1,mxREAL);
	p=mxGetPr(plhs[0]);
	*p=qg;
}


/*
	���ڼ���Qgֵ Qg = ÿ�������ļ������ۼ�
	m:������n���ڵ�Ķ�
	m2: m2=2m=156
	mrows:�ڽӾ�������
	ncols:�ڽӾ�������
	*inMat:ָ���ڽӾ����һ��Ԫ�صĵ�ַ
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
