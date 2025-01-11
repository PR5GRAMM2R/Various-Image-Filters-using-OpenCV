// PSNR.cpp : 콘솔 응용 프로그램에 대한 진입점을 정의합니다.
//

#include "stdafx.h"

#define _USE_MATH_DEFINES
#include <math.h>

//#include <opencv/cv.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

unsigned char getGammaTransformation(unsigned char inputPixel, double gamma);
Point transposition(Point start, Point dest);
Point rotation(Point start, double rot);
unsigned char getBilinearInterpolation(Point pixel, Mat& inputImage);

int main()
{
	double x;

	cout << "감마 값 : ";
	cin >> x;

	Mat img_in;

	// image 읽고 gray로 바꾸기
	img_in = imread("Lena.png");

	//cout << img_in.data[0] << ", " << img_in.data[1] << ", " << img_in.data[2] << endl;

	cvtColor(img_in, img_in, cv::COLOR_RGB2GRAY);

	//cout << img_in.data[0] << ", " << img_in.data[1] << ", " << img_in.data[2] << endl;

	imshow("source img", img_in);

	int input_x_size = img_in.cols;
	int input_y_size = img_in.rows;
	int output_x_size = img_in.cols;
	int output_y_size = img_in.rows;

	Mat img_out(output_y_size, output_x_size, CV_8UC1);

	unsigned char *pData;
	pData = new unsigned char[output_x_size * output_y_size];
	//pData = (unsigned char *)img_in.data;

	Point dest((double)input_x_size / 2, (double)input_y_size / 2);

	///////////////////// 처리하기 ///////////////////
	for (int i = 0; i < output_x_size * output_y_size; i++)	// 예제...
	{
		//int colPosInt = i % output_x_size;	// x 축		// output 의 픽셀 위치 구함
		//int rowPosInt = i / output_x_size;	// y 축		// output 의 픽셀 위치 구함

		pData[i] = getGammaTransformation(img_in.data[i], x);
	}

	img_out.data = (uchar*)pData;
	imshow("output image", img_out);

	waitKey(0);
	
	return 0;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Gamma Transformation 적용
unsigned char getGammaTransformation(unsigned char inputPixel, double gamma)
{
	double input = (double)inputPixel / 255.0;

	double output = input * pow(input, gamma);

	output = output > 1 ? 1 : output;
	output = output < 0 ? 0 : output;

	unsigned char outputPixel = (unsigned char)(output * 255.0);

	return outputPixel;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 픽셀의 평행이동
Point transposition(Point start, Point dest)
{
	return Point(start.x + dest.x, start.y + dest.y);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 픽셀의 회전이동
Point rotation(Point start, double rot)
{
	double x = cos(rot) * start.x - sin(rot) * start.y;
	double y = sin(rot) * start.x + cos(rot) * start.y;

	return Point(x, y);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Bilinear Interpolation 적용
unsigned char getBilinearInterpolation(Point pixel, Mat& inputImage)
{
	int img_x_size = inputImage.cols;
	int img_y_size = inputImage.rows;
	int x_size = inputImage.cols;
	int y_size = inputImage.rows;

	double inputDeltaX = (double)1 / (img_x_size - 1);
	double inputDeltaY = (double)1 / (img_y_size - 1);

	double outputDeltaX = (double)1 / (x_size - 1);
	double outputDeltaY = (double)1 / (y_size - 1);

	double colPosTemp = pixel.x;	// x 축		// output 의 픽셀 위치 구함
	double rowPosTemp = pixel.y;	// y 축		// output 의 픽셀 위치 구함

	double colPos = colPosTemp * outputDeltaX;	// x 축		// input 와 비교한 output 의 픽셀 위치 구함
	double rowPos = rowPosTemp * outputDeltaY;	// y 축

	colPos = (colPos < 0) ? 0 : colPos;		// 가용 픽셀 범위를 벗어나는 접근 방지
	colPos = (colPos > 1) ? 1 : colPos;
	rowPos = (rowPos < 0) ? 0 : rowPos;
	rowPos = (rowPos > 1) ? 1 : rowPos;

	int upLeftColPos = (colPos == 1) ? img_x_size - 1 : (int)trunc(colPos * (img_x_size - 1));	// output 픽셀 주변의 사각형의 좌상단 픽셀 좌표를 구함
	int upLeftRowPos = (rowPos == 1) ? img_y_size - 1 : (int)trunc(rowPos * (img_y_size - 1));
	unsigned int upLeftValue = inputImage.data[upLeftRowPos * img_x_size + upLeftColPos];

	int upRightColPos = upLeftColPos + 1;																					// output 픽셀 주변의 사각형의 우상단 픽셀 좌표를 구함
	int upRightRowPos = upLeftRowPos;
	unsigned int upRightValue = inputImage.data[upRightRowPos * img_x_size + upRightColPos];

	int downLeftColPos = upLeftColPos;																						// output 픽셀 주변의 사각형의 좌하단 픽셀 좌표를 구함
	int downLeftRowPos = upLeftRowPos + 1;
	unsigned int downLeftValue = inputImage.data[downLeftRowPos * img_x_size + downLeftColPos];

	int downRightColPos = upLeftColPos + 1;																					// output 픽셀 주변의 사각형의 우하단 픽셀 좌표를 구함
	int downRIghtRowPos = upLeftRowPos + 1;
	unsigned int downRIghtValue = inputImage.data[downRIghtRowPos * img_x_size + downRightColPos];

	double dx = (colPos - ((double)upLeftColPos * inputDeltaX)) * (img_x_size - 1);
	double dy = (rowPos - ((double)upLeftRowPos * inputDeltaY)) * (img_y_size - 1);

	double sum = (double)upLeftValue * (1 - dx) * (1 - dy) + (double)upRightValue * dx * (1 - dy) + (double)downLeftValue * (1 - dx) * dy + (double)downRIghtValue * dx * dy;

	return (unsigned char)round(sum);// / (imgDeltaX * imgDeltaY));
}