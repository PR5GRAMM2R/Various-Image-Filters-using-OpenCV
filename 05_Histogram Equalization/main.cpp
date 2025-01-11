// PSNR.cpp : 콘솔 응용 프로그램에 대한 진입점을 정의합니다.
//

#include "stdafx.h"

#define _USE_MATH_DEFINES
#include <math.h>

//#include <opencv/cv.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat getHistogramEqualization(Mat& input);
unsigned char getNegativeTransformation(unsigned char inputPixel);
unsigned char getGammaTransformation(unsigned char inputPixel, double gamma);
Point transposition(Point start, Point dest);
Point rotation(Point start, double rot);
unsigned char getBilinearInterpolation(Point pixel, Mat& inputImage);

int main()
{
	/*double x;

	cout << "감마 값 : ";
	cin >> x;*/

	Mat img_in;

	// image 읽고 gray로 바꾸기
	img_in = imread("Lena.png");

	//cout << img_in.data[0] << ", " << img_in.data[1] << ", " << img_in.data[2] << endl;

	Mat original;

	cvtColor(img_in, original, cv::COLOR_RGB2GRAY);

	//cout << img_in.data[0] << ", " << img_in.data[1] << ", " << img_in.data[2] << endl;

	imshow("Original img", original);

	int input_x_size = img_in.cols;
	int input_y_size = img_in.rows;
	int output_x_size = img_in.cols;
	int output_y_size = img_in.rows;

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Step 1 처리하기
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Mat lena1(output_y_size, output_x_size, CV_8UC1);			// lena1 생성

	unsigned char *pData1;
	pData1 = new unsigned char[output_x_size * output_y_size];
	
	///////////////////// 처리하기 ///////////////////
	for (int i = 0; i < output_x_size * output_y_size; i++)
	{
		pData1[i] = (unsigned char)((double)(original.data[i]) / 2.0);				// s = r / 2 의 변환을 통해 lena1 만들기

		pData1[i] = pData1[i] > 255 ? 255 : pData1[i];
		pData1[i] = pData1[i] < 0 ? 0 : pData1[i];
	}

	lena1.data = (uchar*)pData1;
	imshow("lena1 image", lena1);

	Mat lena2(output_y_size, output_x_size, CV_8UC1);			// lena2 생성

	unsigned char *pData2;
	pData2 = new unsigned char[output_x_size * output_y_size];
	
	///////////////////// 처리하기 ///////////////////
	for (int i = 0; i < output_x_size * output_y_size; i++)
	{
		pData2[i] = (unsigned char)(128.0 + (double)(original.data[i]) / 2.0);		// s = 128 + r / 2 의 변환을 통해 lena2 만들기

		pData2[i] = pData2[i] > 255 ? 255 : pData2[i];
		pData2[i] = pData2[i] < 0 ? 0 : pData2[i];
	}

	lena2.data = (uchar*)pData2;
	imshow("lena2 image", lena2);

	//waitKey(0);

	//return 0;

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Step 2 처리하기
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	original = getHistogramEqualization(original);
	lena1 = getHistogramEqualization(lena1);
	lena2 = getHistogramEqualization(lena2);

	imshow("Original img", original);
	imshow("lena1 image", lena1);
	imshow("lena2 image", lena2);

	waitKey(0);
	
	return 0;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Histogram Equalization 적용
Mat getHistogramEqualization(Mat& input)
{
	int output_x_size = input.cols;
	int output_y_size = input.rows;
	
	int L = 256;		// 8 bit 이므로 L 은 256

	int n[256] = { 0 };			// 한 픽셀에 256 가지의 값들 중 하나를 가질 수 있으므로, 각 값마다의 픽셀 수를 배열로서 저장

	for (int i = 0; i < output_x_size * output_y_size; i++) {
		n[input.data[i]]++;
	}

	double p[256] = { 0.0 };		// 전체 픽셀 개수들 기준으로 각 값마다의 픽셀 수를 비율로 저장

	for (int i = 0; i < L; i++) {
		p[i] = (double)(n[i]) / (double)(output_x_size * output_y_size);
	}

	int s[256] = { 0 };				// 처리 전의 픽셀값이 Histogram Equalization 을 통해 어떤 픽셀값으로 바뀌는 지 저장

	for (int i = 0; i < L; i++) {
		double temp = 0;

		for (int j = 0; j < i + 1; j++) {
			temp += p[j];
		}

		s[i] = round((double)(L - 1) * temp);		// 결과값을 반올림하여 s 배열에 저장
	}

	Mat output(output_y_size, output_x_size, CV_8UC1);
	
	unsigned char* pData = new unsigned char[output_x_size * output_y_size];

	for (int i = 0; i < output_x_size * output_y_size; i++) {
		pData[i] = s[input.data[i]];			// s 배열을 참조하여 각 픽셀값들을 변환해줌
	}

	output.data = pData;

	return output;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Negative Transformation  적용
unsigned char getNegativeTransformation(unsigned char inputPixel)
{
	return 255 - inputPixel;
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