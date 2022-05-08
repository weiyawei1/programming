// pyCall.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int fac(int n);



int fac(int n)
{
	if (n<2) 
		return (1);
	return (n)*fac(n-1);
}



int main(int argc, _TCHAR* argv[])
{
	printf("%d",fac(10));
	getchar();
	return 0;
}