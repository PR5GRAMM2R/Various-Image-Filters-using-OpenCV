// PSNR.cpp : 콘솔 응용 프로그램에 대한 진입점을 정의합니다.
//

#include "stdafx.h"			// 2020203011 배주환

#define _USE_MATH_DEFINES
#include <math.h>
#include <random>
#include <vector>

//#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
//#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;

Mat medianFilter3x3(Mat& inputImage, Size size);
int pixelAdd(int pixel1, int pixel2);
int pixelSub(int pixel1, int pixel2);
Mat applyLinearFilter(Mat& ori, int* f, int row, int col);
Mat getHistogramEqualization(Mat& input);
unsigned char getNegativeTransformation(unsigned char inputPixel);
unsigned char getGammaTransformation(unsigned char inputPixel, double gamma);
Point2f transposition(Point2f start, Point2f dest);
Point rotation(Point start, double rot);
Point rotation(Point start, Mat2f rotM);
unsigned char getBilinearInterpolation(Point pixel, Mat& inputImage);
void getBilinearInterpolation(Point2f inputPixel, Point outputPixel, Mat& inputImage, Mat& outputImage);


int main()
{
	/*double x;

	cout << "감마 값 : ";
	cin >> x;*/

	Mat img_in;

	// image 읽고 gray로 바꾸기
	img_in = imread("lena_t.jpg", IMREAD_GRAYSCALE);
	//img_in = imread("Lena.png");

	if (img_in.empty()) {
		cout << "No Image" << endl;
		return 0;
	}

	//cout << img_in.data[0] << ", " << img_in.data[1] << ", " << img_in.data[2] << endl;

	Mat original(img_in);

	//cout << img_in.data[0] << ", " << img_in.data[1] << ", " << img_in.data[2] << endl;

	imshow("input image", img_in);
	//waitKey(0);
	Size size = img_in.size();

	vector<vector<float>> input = {
		{ 173,		284,		0,			0,		1,	0 },
		{ 0,		0,			173,		284,	0,	1 },
		{ 477,		33,			0,			0,		1,	0 },
		{ 0,		0,			477,		33,		0,	1 },
		{ 248,		455,		0,			0,		1,	0 },
		{ 0,		0,			248,		455,	0,	1 },
		{ 553,		193,		0,			0,		1,	0 },
		{ 0,		0,			553,		193,	0,	1 },
	};

	/*vector<vector<float>> input = {
		{ 100,		100,		0,			0,		1,	0 },
		{ 0,		0,			100,		100,	0,	1 },
		{ 412,		100,		0,			0,		1,	0 },
		{ 0,		0,			412,		100,	0,	1 },
		{ 100,		412,		0,			0,		1,	0 },
		{ 0,		0,			100,		412,	0,	1 },
		{ 412,		412,		0,			0,		1,	0 },
		{ 0,		0,			412,		412,	0,	1 },
	};*/

	Mat inputM(8, 6, CV_32F);

	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 6; j++) {
			inputM.at<float>(i, j) = input[i][j];
		}
	}

	/*for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 6; j++) {
			cout << inputM.at<float>(i, j) << " ";
		}
		cout << endl;
	}
	cout << endl << endl;*/

	Mat inputInverseM(6, 8, CV_32F);

	invert(inputM, inputInverseM, DECOMP_SVD);

	/*for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 8; j++) {
			cout << inputInverseM.at<float>(i, j) << " ";
		}
		cout << endl;
	}
	cout << endl << endl;*/

	vector<float> outputTemp = { 100, 100, 412, 100, 100, 412, 412, 412 };

	//vector<float> outputTemp = { 173, 284, 477, 33, 248, 455, 553, 193 };

	vector<float> tempInverseM;

	for (int i = 0; i < 6; i++) {
		float sum = 0;

		for (int j = 0; j < 8; j++) {
			sum += inputInverseM.at<float>(i, j) * outputTemp[j];
		}

		tempInverseM.emplace_back(sum);
	}

	cout << "m1 : " << tempInverseM[0] << endl;
	cout << "m2 : " << tempInverseM[1] << endl;
	cout << "m3 : " << tempInverseM[2] << endl;
	cout << "m4 : " << tempInverseM[3] << endl;
	cout << "t1 : " << tempInverseM[4] << endl;
	cout << "t2 : " << tempInverseM[5] << endl;
	cout << endl << endl;

	//Mat inverseM(2, 2, CV_32F);
	Mat inverseM(3, 3, CV_32F);

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			inverseM.at<float>(i, j) = tempInverseM[2 * i + j];
		}
	}

	inverseM.at<float>(0, 2) = tempInverseM[4];
	inverseM.at<float>(1, 2) = tempInverseM[5];
	inverseM.at<float>(2, 2) = 1;

	//Mat M(2, 2, CV_32F);
	Mat M(3, 3, CV_32F);
	//Point2f T;

	invert(inverseM, M, DECOMP_SVD);
	//T = Point2f(-tempInverseM[4], -tempInverseM[5]);
	//T = Point2f(tempInverseM[4], tempInverseM[5]);

	cout << endl << endl;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			cout << inverseM.at<float>(i, j) << " ";
		}
		cout << endl;
	}

	Mat output(Size(512, 512), CV_8U);

	for (int y = 0; y < output.rows; y++) {
		for (int x = 0; x < output.cols; x++) {
	//for (int y = 0; y < size.height; y++) {
	//	for (int x = 0; x < size.width; x++) {
			Point2f p(x, y);
			double tempX;
			double tempY;

			//p = transposition(p, T);

			tempX = M.at<float>(0, 0) * p.x + M.at<float>(0, 1) * p.y + M.at<float>(0, 2);
			tempY = M.at<float>(1, 0) * p.x + M.at<float>(1, 1) * p.y + M.at<float>(1, 2);
			//tempX = inverseM.at<float>(0, 0) * p.x + inverseM.at<float>(0, 1) * p.y;
			//tempY = inverseM.at<float>(1, 0) * p.x + inverseM.at<float>(1, 1) * p.y;
			p = Point2f(tempX, tempY);
			
			//p = transposition(p, T);

			if (!(p.x < 0 || p.x > size.width || p.y < 0 || p.y > size.height))
			//if (!(p.x < 0 || p.x > output.cols || p.y < 0 || p.y > output.rows))
				//output.data[(int)p.y * output.cols + (int)p.x] = original.data[y * size.width + x];
				//output.at<unsigned char>(y, x) = 128;
				getBilinearInterpolation(p, Point(x, y), original, output);

			//getBilinearInterpolation(p, Point(x, y), original, output);

			//output.at<unsigned char>(y, x) = getBilinearInterpolation(p, original);

			//cout << x << " " << y << " " <<p.x << " " << p.y << endl;
		}
	}

	//imshow("original image", original);
	imshow("output image", output);

	waitKey(0);

	

	return 0;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 픽셀의 평행이동
Point2f transposition(Point2f start, Point2f dest)
{
	return Point2f(start.x + dest.x, start.y + dest.y);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Bilinear Interpolation 적용
void getBilinearInterpolation(Point2f inputPixel, Point outputPixel, Mat& inputImage, Mat& outputImage)
{
	int input_x_size = inputImage.cols;
	int input_y_size = inputImage.rows;
	int output_x_size = outputImage.cols;
	int output_y_size = outputImage.rows;

	double inputDeltaX = (double)1 / (input_x_size - 1);
	double inputDeltaY = (double)1 / (input_y_size - 1);

	//inputImage.data[(int)inputPixel.x * input_x_size + (int)inputPixel.y] = 255;
	//outputImage.data[outputPixel.y * output_x_size + outputPixel.x] = inputImage.data[(int)inputPixel.x * input_x_size + (int)inputPixel.y];
	//	//at<unsigned char>(outputPixel.y, outputPixel.x) = sum;
	//return;

	//double outputDeltaX = (double)1 / (output_x_size - 1);
	//double outputDeltaY = (double)1 / (output_y_size - 1);

	double colPos = inputPixel.x * inputDeltaX;	// x 축		// input 와 비교한 output 의 픽셀 위치 구함
	double rowPos = inputPixel.y * inputDeltaY;	// y 축

	colPos = (colPos < 0) ? 0 : colPos;		// 가용 픽셀 범위를 벗어나는 접근 방지
	colPos = (colPos > 1) ? 1 : colPos;
	rowPos = (rowPos < 0) ? 0 : rowPos;
	rowPos = (rowPos > 1) ? 1 : rowPos;

	int upLeftColPos = (colPos == 1) ? input_x_size - 1 : (int)trunc(colPos * (input_x_size - 1));	// output 픽셀 주변의 사각형의 좌상단 픽셀 좌표를 구함
	int upLeftRowPos = (rowPos == 1) ? input_y_size - 1 : (int)trunc(rowPos * (input_y_size - 1));
	unsigned int upLeftValue = inputImage.data[upLeftRowPos * input_x_size + upLeftColPos];

	int upRightColPos = upLeftColPos + 1;																					// output 픽셀 주변의 사각형의 우상단 픽셀 좌표를 구함
	int upRightRowPos = upLeftRowPos;
	unsigned int upRightValue = inputImage.data[upRightRowPos * input_x_size + upRightColPos];

	int downLeftColPos = upLeftColPos;																						// output 픽셀 주변의 사각형의 좌하단 픽셀 좌표를 구함
	int downLeftRowPos = upLeftRowPos + 1;
	unsigned int downLeftValue = inputImage.data[downLeftRowPos * input_x_size + downLeftColPos];

	int downRightColPos = upLeftColPos + 1;																					// output 픽셀 주변의 사각형의 우하단 픽셀 좌표를 구함
	int downRIghtRowPos = upLeftRowPos + 1;
	unsigned int downRIghtValue = inputImage.data[downRIghtRowPos * input_x_size + downRightColPos];

	double dx = (colPos - ((double)upLeftColPos * inputDeltaX)) *(input_x_size - 1);
	double dy = (rowPos - ((double)upLeftRowPos * inputDeltaY)) *(input_y_size - 1);

	double sum = (double)upLeftValue * (1 - dx) * (1 - dy) + (double)upRightValue * dx * (1 - dy) + (double)downLeftValue * (1 - dx) * dy + (double)downRIghtValue * dx * dy;

	//inputImage.data[upLeftRowPos * input_x_size + upLeftColPos] = 255;
	outputImage.data[outputPixel.y * output_x_size + outputPixel.x] = (unsigned char)sum;
		//at<unsigned char>(outputPixel.y, outputPixel.x) = sum;
	return;
	//return (unsigned char)round(sum);// / (imgDeltaX * imgDeltaY));
}