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

double Qov(double *inMatX,double *inMat,size_t Kq,double *m,size_t Mq,size_t mrowsX,size_t ncolsX,size_t mrows,size_t ncols);//计算qov值
void find(size_t mrows,size_t k,double *inPtr,size_t *oPtr,size_t *nk); //寻找矩阵inPtr中第K列的非零值，将结果返回给向量oPtr
double wSum(size_t mrows,size_t m, size_t n,double *ptr); //计算邻接矩阵的第m行的n个元素之和
double lSum(size_t mrows,size_t m, size_t n,double *ptr); //计算邻接矩阵的第n列的m个元素之和

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	//nlhs：输出参数个数
	//plhs：输出参数列表
	//nrhs：输入参数个数
	//prhs：输入参数列表   

	// Check for proper number of input and output arguments 
	if (nrhs != 5) {
		mexErrMsgIdAndTxt("MATLAB:mxcreatecellmatrix:nrhs",
			"输入参数个数不正确");
	}
	if (nlhs != 1) {
		mexErrMsgIdAndTxt("MATLAB:mxcreatecellmatrix:nlhs",
			"输出参数个数不正确");
	}

	//第一个输入变量的传递
	size_t mrowsX;    //行数
	size_t ncolsX;    //列数
	//mxArray *inMat;  //接收输入参数的指针
	double *inMatX;
	mrowsX = mxGetM(prhs[0]); //获取矩阵行数
	ncolsX = mxGetN(prhs[0]); //获取矩阵列数
	inMatX = mxGetPr(prhs[0]);//获取输入矩阵的指针

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
	size_t K = (size_t)mxGetScalar(prhs[2]);

	//第四个输入变量的传递
	double  *m = mxGetPr(prhs[3]);

	//第五个输入变量的传递
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
	用于计算Qov值
	M:邻接矩阵边数
	n:脑网络代数
	mrows:邻接矩阵行数
	ncols:邻接矩阵列数
	*inMat:指向邻接矩阵第一个元素的地址
*/
double Qov(double *inMatX,double *inMat,size_t Kq,double *m,size_t Mq,size_t mrowsX,size_t ncolsX,size_t mrows,size_t ncols)
{
	double qovTemp=0;   

	for(size_t k=0;k<Kq;k++){
		size_t *pointk = (size_t *)malloc(mrowsX*sizeof(size_t));//第K个社区中所有节点标号
        //初始化内存
        for (size_t i=0;i<mrowsX;i++){
			pointk[i]=0;
		}
		size_t nk=0;
		find(mrowsX,k,inMatX,pointk,&nk);//求出第k个社区中所有节点标号,nk为社区规模
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

//寻找矩阵inPtr中第k列的非零值的索引index，将结果返回给向量oPtr
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

//计算邻接矩阵的第m行的n个元素之和
double wSum(size_t mrows,size_t m,size_t n, double *ptr)
{
	double sum=0;
	for(size_t i=0;i<n;i++){
		sum = sum + ptr[i*mrows+m-1];
	}
	return sum;
}

//计算邻接矩阵的第n列的m个元素之和
double lSum(size_t mrows,size_t m,size_t n, double *ptr)
{
	double sum=0;
	for(size_t i=0;i<m;i++){
		sum = sum + ptr[n*mrows+i];
	}
	return sum;
}
