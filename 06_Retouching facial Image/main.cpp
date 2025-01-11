// PSNR.cpp : 콘솔 응용 프로그램에 대한 진입점을 정의합니다.
//

#include "stdafx.h"			// 2020203011 배주환

#define _USE_MATH_DEFINES
#include <math.h>

//#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
//#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;

Mat applyLinearFilter(Mat& ori, Mat& f);
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
	img_in = imread("Sample.png");

	if (img_in.empty()) {
		cout << "No Image" << endl;
		return 0;
	}

	//cout << img_in.data[0] << ", " << img_in.data[1] << ", " << img_in.data[2] << endl;

	Mat original(img_in);

	//cout << img_in.data[0] << ", " << img_in.data[1] << ", " << img_in.data[2] << endl;

	imshow("Original img", original);
	//waitKey(0);
	Size size = img_in.size();

	//cout << size.height << ", " << size.width << endl;

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Step 1 처리하기
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Mat facePixels(size, CV_8UC3, Scalar(0, 0, 0));		// 원본에서 얼굴 정보만 저장할 이미지를 생성

	Mat faceMap(size, CV_8UC1, Scalar(0, 0, 0));		// 원본에서 얼굴에 속하는 부분만 255 를 저장하는 이미지를 생성

	/*for (int i = 0; i < 100; i++) {
		for (int j = 0; j < size.width; j++) {
			facePixels.at<Vec3b>(Point(i, j)) = Vec3b(255, 0, 0);
		}
	}*/

	int sumB = 0;		// 추출한 얼굴의 평균적인 픽셀값을 만들기 위한 변수들
	int sumG = 0;
	int sumR = 0;
	int num = 0;

	for (int i = 0; i < size.width; i++) {
		for (int j = 0; j < size.height; j++) {
			int b = original.at<Vec3b>(Point(i, j)).val[0];
			int g = original.at<Vec3b>(Point(i, j)).val[1];
			int r = original.at<Vec3b>(Point(i, j)).val[2];

			if (b >= 20 && b <= 175 && g >= 25 && g <= 190 && r >= 180 && r <= 230) {	// 해당 BGR 조건에 부합하는 원본 픽셀에 대해
				faceMap.at<uchar>(Point(i, j)) = 255;									//	faceMap 의 해당 픽셀에 255 를 저장하고,
				facePixels.at<Vec3b>(Point(i, j)) = Vec3b(b, g, r);						//	facePixels 이미지에 원본 픽셀을 저장

				sumB += b;
				sumG += g;
				sumR += r;
				num++;
			}
		}
	}

	imshow("facePixels", facePixels);
	imshow("faceMap", faceMap);


	Vec3b temp = Vec3b((float)sumB / num, (float)sumG / num, (float)sumR / num);		// 추출한 얼굴의 평균적인 픽셀값을 만듦

	//cout << (int)temp.val[0] << ", " << (int)temp.val[1] << ", " << (int)temp.val[2] << endl;

	for (int i = 0; i < size.width; i++) {
		for (int j = 0; j < size.height; j++) {
			int b = facePixels.at<Vec3b>(Point(i, j)).val[0];
			int g = facePixels.at<Vec3b>(Point(i, j)).val[1];
			int r = facePixels.at<Vec3b>(Point(i, j)).val[2];

			if (b == 0 && g == 0 && r == 0) {
				facePixels.at<Vec3b>(Point(i, j)) = temp;			// 부드러운 필터링 효과를 위해 facePixels 에서 픽셀값이 0 인 픽셀은
			}														//	얼굴의 평균적인 픽셀값으로 바꿈
		}
	}

	//imshow("facePixels!!!!!!!", facePixels);
	//waitKey();

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Step 2 처리하기
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Mat movingAverageFilter(5, 5, CV_8S);						// Moving Average Filter 을 위한 5 x 5 필터

	uchar pData[25];
	for (int i = 0; i < 25; i++) pData[i] = 1;

	//for (int i = 0; i < 121; i++) cout << (int)pData[i];

	movingAverageFilter.data = pData;

	Mat filterApplied(applyLinearFilter(facePixels, movingAverageFilter));	// Smooth Filter 적용

	//imshow("filterApplied", filterApplied);
	//waitKey();

	Mat result(size, CV_8UC3);

	for (int i = 0; i < size.width; i++) {
		for (int j = 0; j < size.height; j++) {									// 원본과 필터링한 이미지를 합성
			if (faceMap.at<uchar>(Point(i, j)) == 255) {
				result.at<Vec3b>(Point(i, j)) = filterApplied.at<Vec3b>(Point(i, j));
			}
			else {
				result.at<Vec3b>(Point(i, j)) = original.at<Vec3b>(Point(i, j));
			}
		}
	}

	imshow("result", result);
	waitKey();
	
	return 0;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Smoothing Filter 적용
Mat applyLinearFilter(Mat& ori, Mat& f)		// linear Filter 적용하는 함수
{
	Size oSize = ori.size();
	Size fSize = f.size();

	Mat filterApplied(oSize, CV_8UC3);
	
	int fSum = 0;

	for (int i = 0; i < fSize.width * fSize.height; i++) {		// 필터의 각 요소의 합을 구함
		fSum += f.data[i];
	}

	// cout << fSum << endl;

	for (int i = 0; i < oSize.width - fSize.width; i++) {			// 필터를 적용
		for (int j = 0; j < oSize.height - fSize.height; j++) {
			int sumB = 0;
			int sumG = 0;
			int sumR = 0;

			for (int x = 0; x < fSize.width; x++) {
				for (int y = 0; y < fSize.height; y++) {
					sumB += (int)ori.at<Vec3b>(Point(i + x, j + y)).val[0] * (int)f.at<uchar>(Point(x, y));
					sumG += (int)ori.at<Vec3b>(Point(i + x, j + y)).val[1] * (int)f.at<uchar>(Point(x, y));
					sumR += (int)ori.at<Vec3b>(Point(i + x, j + y)).val[2] * (int)f.at<uchar>(Point(x, y));
				}
			}

			filterApplied.at<Vec3b>(Point(i + (fSize.width - 1) / 2, j + (fSize.height - 1) / 2)) = Vec3b((float)sumB / fSum, (float)sumG / fSum, (float)sumR / fSum);
		}
	}

	return filterApplied;
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