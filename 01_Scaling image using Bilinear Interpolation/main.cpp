// PSNR.cpp : �ܼ� ���� ���α׷��� ���� �������� �����մϴ�.
//

#include "stdafx.h"		// 2020203011  ����ȯ
#include <math.h>

//#include <opencv/cv.h>
#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;

int x_size = 512;
int y_size = 512;

int main()
{
	Mat img_in;
	Mat img_out(y_size, x_size, CV_8UC1);

	// image �а� gray�� �ٲٱ�
	img_in = imread("Lena_256x256.png");
	cvtColor(img_in, img_in, cv::COLOR_RGB2GRAY);
	imshow("source img", img_in);

	int img_x_size = img_in.cols;
	int img_y_size = img_in.rows;

	unsigned char *pData;
	pData = new unsigned char[x_size * y_size];
	//pData = (unsigned char *)img_in.data;

	double inputDeltaX = (double)1 / (img_x_size - 1);
	double inputDeltaY = (double)1 / (img_y_size - 1);

	double outputDeltaX = (double)1 / (x_size - 1);
	double outputDeltaY = (double)1 / (y_size - 1);

	///////////////////// ó���ϱ� ///////////////////
	for (int i = 0; i < x_size * y_size; i++)	// ����...
	{
		int colPosInt = i % x_size;	// x ��		// output �� �ȼ� ��ġ ����
		int rowPosInt = i / x_size;	// y ��		// output �� �ȼ� ��ġ ����

		double colPos = (double)colPosInt * outputDeltaX;	// x ��		// input �� ���� output �� �ȼ� ��ġ ����
		double rowPos = (double)rowPosInt * outputDeltaY;	// y ��

		colPos = (colPos < 0) ? 0 : colPos;		// ���� �ȼ� ������ ����� ���� ����
		colPos = (colPos > 1) ? 1 : colPos;
		rowPos = (rowPos < 0) ? 0 : rowPos;
		rowPos = (rowPos > 1) ? 1 : rowPos;

		int upLeftColPos = (colPos == 1) ? img_x_size - 1 : (int)trunc(colPos * (img_x_size - 1));	// output �ȼ� �ֺ��� �簢���� �»�� �ȼ� ��ǥ�� ����
		int upLeftRowPos = (rowPos == 1) ? img_y_size - 1 : (int)trunc(rowPos * (img_y_size - 1));
		unsigned int upLeftValue = img_in.data[upLeftRowPos * img_x_size + upLeftColPos];

		int upRightColPos = upLeftColPos + 1;																					// output �ȼ� �ֺ��� �簢���� ���� �ȼ� ��ǥ�� ����
		int upRightRowPos = upLeftRowPos;
		unsigned int upRightValue = img_in.data[upRightRowPos * img_x_size + upRightColPos];

		int downLeftColPos = upLeftColPos;																						// output �ȼ� �ֺ��� �簢���� ���ϴ� �ȼ� ��ǥ�� ����
		int downLeftRowPos = upLeftRowPos + 1;
		unsigned int downLeftValue = img_in.data[downLeftRowPos * img_x_size + downLeftColPos];

		int downRightColPos = upLeftColPos + 1;																					// output �ȼ� �ֺ��� �簢���� ���ϴ� �ȼ� ��ǥ�� ����
		int downRIghtRowPos = upLeftRowPos + 1;
		unsigned int downRIghtValue = img_in.data[downRIghtRowPos * img_x_size + downRightColPos];

		double dx = (colPos - ((double)upLeftColPos * inputDeltaX)) * (img_x_size - 1);
		double dy = (rowPos - ((double)upLeftRowPos * inputDeltaY)) * (img_y_size - 1);

		double sum = (double)upLeftValue * (1 - dx) * (1 - dy) + (double)upRightValue * dx * (1 - dy) + (double)downLeftValue * (1 - dx) * dy + (double)downRIghtValue * dx * dy;

		pData[i] = (unsigned char)round(sum);// / (imgDeltaX * imgDeltaY));
	}

	img_out.data = (uchar*)pData;
	imshow("output image", img_out);

	waitKey(0);

	return 0;
}