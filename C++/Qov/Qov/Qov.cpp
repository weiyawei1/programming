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

double Qov(double *inMatX,double *inMat,size_t Kq,double *m,size_t Mq,size_t mrowsX,size_t ncolsX,size_t mrows,size_t ncols);//����qovֵ
void find(size_t mrows,size_t k,double *inPtr,size_t *oPtr,size_t *nk); //Ѱ�Ҿ���inPtr�е�K�еķ���ֵ����������ظ�����oPtr
double wSum(size_t mrows,size_t m, size_t n,double *ptr); //�����ڽӾ���ĵ�m�е�n��Ԫ��֮��
double lSum(size_t mrows,size_t m, size_t n,double *ptr); //�����ڽӾ���ĵ�n�е�m��Ԫ��֮��

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	//nlhs�������������
	//plhs����������б�
	//nrhs�������������
	//prhs����������б�   

	// Check for proper number of input and output arguments 
	if (nrhs != 5) {
		mexErrMsgIdAndTxt("MATLAB:mxcreatecellmatrix:nrhs",
			"���������������ȷ");
	}
	if (nlhs != 1) {
		mexErrMsgIdAndTxt("MATLAB:mxcreatecellmatrix:nlhs",
			"���������������ȷ");
	}

	//��һ����������Ĵ���
	size_t mrowsX;    //����
	size_t ncolsX;    //����
	//mxArray *inMat;  //�������������ָ��
	double *inMatX;
	mrowsX = mxGetM(prhs[0]); //��ȡ��������
	ncolsX = mxGetN(prhs[0]); //��ȡ��������
	inMatX = mxGetPr(prhs[0]);//��ȡ��������ָ��

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
	size_t K = (size_t)mxGetScalar(prhs[2]);

	//���ĸ���������Ĵ���
	double  *m = mxGetPr(prhs[3]);

	//�������������Ĵ���
	size_t M = (size_t)mxGetScalar(prhs[4]); 
    
   	double qov=0;
  
	qov = Qov(inMatX,inMat,K,m,M,mrowsX,ncolsX,mrows,ncols);
	//printf("qov= %f",qov);
	double *p;
	plhs[0]=mxCreateDoubleMatrix(1,1,mxREAL);
	p=mxGetPr(plhs[0]);
	*p=qov;
}


/*
	���ڼ���Qovֵ
	M:�ڽӾ������
	n:���������
	mrows:�ڽӾ�������
	ncols:�ڽӾ�������
	*inMat:ָ���ڽӾ����һ��Ԫ�صĵ�ַ
*/
double Qov(double *inMatX,double *inMat,size_t Kq,double *m,size_t Mq,size_t mrowsX,size_t ncolsX,size_t mrows,size_t ncols)
{
	double qovTemp=0;   

	for(size_t k=0;k<Kq;k++){
		size_t *pointk = (size_t *)malloc(mrowsX*sizeof(size_t));//��K�����������нڵ���
        //��ʼ���ڴ�
        for (size_t i=0;i<mrowsX;i++){
			pointk[i]=0;
		}
		size_t nk=0;
		find(mrowsX,k,inMatX,pointk,&nk);//�����k�����������нڵ���,nkΪ������ģ
		//r//r[]=array[][]
		double *r=(double*)malloc(nk*nk*sizeof(double));
		for (size_t i=0;i<nk*nk;i++){
			r[i]=0;
		}
		for (size_t i=0;i<nk;i++){
			for(size_t j=0;j<nk;j++){
				size_t iw = pointk[i];
				size_t jw = pointk[j];
				double fuic =60 * inMatX[k*mrowsX + iw]-30;
				double fujc=60*inMatX[k*mrowsX+jw]-30;
				r[j*mrowsX+i] = 1/(1+exp(-fuic)*(1+exp(-fujc)));
			}
		}
		//w
		double *w;//w[]=array[][]
		w = (double *)malloc(nk * nk * sizeof(double));
		for (size_t i=0;i<nk*nk;i++){
			w[i]=0.0000;
		}
		for (size_t i=0;i<nk;i++){
			for(size_t j=0;j<nk;j++){
				// w(i,j)=sum(R(i,:)).*sum(R(:,j))./nk.^2;
				w[j*mrowsX+i] = wSum(nk,i,nk,w) * lSum(nk,nk,j,w) / pow((double)nk,2);
			}
		}
      
		//Qov
		for (size_t i=0;i<nk;i++){
			for(size_t j=0;j<nk;j++){
				//Qov = Qov + (R(i,j).*W(pointk(i),pointk(j)) - w(i,j).*(m(pointk(i)).*m(pointk(j)))./(2*Mq));
				size_t iw = pointk[i];
				size_t jw = pointk[j];
				qovTemp = qovTemp + (r[j*nk+i] * inMat[jw*nk+iw] - w[j*nk+i] * (m[iw] * m[jw])/(Mq<<1));
			}
		}
		free(r);
		free(w);
 		free(pointk);
	}
 
    
//     printf("Mq= %d",Mq);
	return qovTemp / (Mq<<1);
}

//Ѱ�Ҿ���inPtr�е�k�еķ���ֵ������index����������ظ�����oPtr
void find(size_t mrows,size_t k,double *inPtr,size_t *oPtr,size_t *nk)
{	

	size_t r=0;
    for(size_t i=0;i<mrows;i++){
		double temp = inPtr[(k)*mrows+i];
		if(temp > 0.0 || temp < 0.0){
            oPtr[r]=i;
			r++;
		}
	}
	*nk=r;
	
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

//�����ڽӾ���ĵ�n�е�m��Ԫ��֮��
double lSum(size_t mrows,size_t m,size_t n, double *ptr)
{
	double sum=0;
	for(size_t i=0;i<m;i++){
		sum = sum + ptr[n*mrows+i];
	}
	return sum;
}
