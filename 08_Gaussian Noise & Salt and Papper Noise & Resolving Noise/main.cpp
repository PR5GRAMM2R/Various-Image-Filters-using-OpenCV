// PSNR.cpp : 콘솔 응용 프로그램에 대한 진입점을 정의합니다.
//

#include "stdafx.h"			// 2020203011 배주환

#define _USE_MATH_DEFINES
#include <math.h>
#include <random>

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

	Mat gaussianNoiseLow(size.height, size.width, CV_8UC3);
	Mat gaussianNoiseHigh(size.height, size.width, CV_8UC3);

	Mat impulseNoiseLow(size.height, size.width, CV_8UC3);
	Mat impulseNoiseHigh(size.height, size.width, CV_8UC3);
	

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Gaussian noise 영상 생성
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	Mat gaussianNoise(size.height, size.width, CV_8UC3);

	cv::randn(gaussianNoise, 0, 20);

	gaussianNoiseLow = original + gaussianNoise;

	/*for (int y = 0; y < size.height; y++) {
		for (int x = 0; x < size.width; x++) {
			int r = gaussianNoiseLow.at<Vec3b>(x, y).val[0];
			int g = gaussianNoiseLow.at<Vec3b>(x, y).val[1];
			int b = gaussianNoiseLow.at<Vec3b>(x, y).val[2];

			gaussianNoiseLow.at<Vec3b>(x, y).val[0] = pixelAdd(r, 0);
		}
	}*/

	cv::randn(gaussianNoise, 0, 100);

	gaussianNoiseHigh = original + gaussianNoise;

	imshow("gaussianNoiseLow", gaussianNoiseLow);
	imshow("gaussianNoiseHigh", gaussianNoiseHigh);
	//waitKey();

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Salt and Pepper noise 영상 생성
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//std::random_device rd1;  // 시드로 사용할 난수 생성기
    std::mt19937 gen1(4567); // Mersenne Twister 엔진을 이용한 난수 생성기
    std::uniform_int_distribution<> dis1(0, 99); 

	//std::random_device rd2;  // 시드로 사용할 난수 생성기
	std::mt19937 gen2(1234); // Mersenne Twister 엔진을 이용한 난수 생성기
	std::uniform_int_distribution<> dis2(0, 99);

	for (int y = 0; y < size.height; y++) {
		for (int x = 0; x < size.width; x++) {
			int temp1 = dis1(gen1);
			if (temp1 % 20 == 0)
				impulseNoiseLow.at<Vec3b>(x, y) = Vec3b(0, 0, 0);
			else if (temp1 % 20 == 1)
				impulseNoiseLow.at<Vec3b>(x, y) = Vec3b(255, 255, 255);
			else {
				impulseNoiseLow.at<Vec3b>(x, y) = original.at<Vec3b>(x, y);
			}

			int temp2 = dis2(gen2);
			if (temp2 % 10 == 0)
				impulseNoiseHigh.at<Vec3b>(x, y) = Vec3b(0, 0, 0);
			else if (temp2 % 10 == 1)
				impulseNoiseHigh.at<Vec3b>(x, y) = Vec3b(255, 255, 255);
			else {
				impulseNoiseHigh.at<Vec3b>(x, y) = original.at<Vec3b>(x, y);
			}
		}
	}

	imshow("impulseNoiseLow", impulseNoiseLow);
	imshow("impulseNoiseHigh", impulseNoiseHigh);


	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//  3 x 3 Median Filter
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Mat gaussianNoiseLowMedian(medianFilter3x3(gaussianNoiseLow, size));
	Mat gaussianNoiseHighMedian(medianFilter3x3(gaussianNoiseHigh, size));
	Mat impulseNoiseLowMedian(medianFilter3x3(impulseNoiseLow, size));
	Mat impulseNoiseHighMedian(medianFilter3x3(impulseNoiseHigh, size));


	imshow("gaussianNoiseLowMedian", gaussianNoiseLowMedian);
	imshow("gaussianNoiseHighMedian", gaussianNoiseHighMedian);
	imshow("impulseNoiseLowMedian", impulseNoiseLowMedian);
	imshow("impulseNoiseHighMedian", impulseNoiseHighMedian);
	//waitKey();


	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Moving Average ( 3 x 3  크기 ) 적용하기 ( 3 x 3 Mean Filter )
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	int movingMeanFilter[3 * 3]  = {1, 1, 1,				// Moving Average Filter 을 위한 3 x 3 필터
									1, 1, 1,
									1, 1, 1 };

	Mat gaussianNoiseLowMean(applyLinearFilter(gaussianNoiseLow, movingMeanFilter, 3, 3));	// 필터 적용하기
	Mat gaussianNoiseHighMean(applyLinearFilter(gaussianNoiseHigh, movingMeanFilter, 3, 3));	// 필터 적용하기
	Mat impulseNoiseLowMean(applyLinearFilter(impulseNoiseLow, movingMeanFilter, 3, 3));	// 필터 적용하기
	Mat impulseNoiseHighMean(applyLinearFilter(impulseNoiseHigh, movingMeanFilter, 3, 3));	// 필터 적용하기

	imshow("gaussianNoiseLowMean", gaussianNoiseLowMean);
	imshow("gaussianNoiseHighMean", gaussianNoiseHighMean);
	imshow("impulseNoiseLowMean", impulseNoiseLowMean);
	imshow("impulseNoiseHighMean", impulseNoiseHighMean);
	waitKey();
	
	return 0;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//  3 x 3 Median Filter
Mat medianFilter3x3(Mat& inputImage, Size size) {
	Mat outputImage(size.height, size.width, CV_8UC3);

	for (int y = 1; y < size.height - 1; y++) {
		for (int x = 1; x < size.width - 1; x++) {
			priority_queue<int, std::vector<int>, std::greater<int>> medianR;
			priority_queue<int, std::vector<int>, std::greater<int>> medianG;
			priority_queue<int, std::vector<int>, std::greater<int>> medianB;

			for (int j = -1; j <= 1; j++) {
				for (int i = -1; i <= 1; i++) {
					medianB.push(inputImage.at<Vec3b>(x + i, y + j).val[0]);
					medianG.push(inputImage.at<Vec3b>(x + i, y + j).val[1]);
					medianR.push(inputImage.at<Vec3b>(x + i, y + j).val[2]);
				}
			}

			for (int i = 0; i < 4; i++) {
				//cout << medianR.top() << " ";
				medianR.pop();
				medianG.pop();
				medianB.pop();
			}
			//cout << endl;

			outputImage.at<Vec3b>(x, y) = Vec3b(medianB.top(), medianG.top(), medianR.top());

			/*
			int median[9];

			for (int j = -1; j <= 1; j++) {
				for (int i = -1; i <= 1; i++) {
					int r = inputImage.at<Vec3b>(x + i, y + j).val[0];
					int g = inputImage.at<Vec3b>(x + i, y + j).val[1];
					int b = inputImage.at<Vec3b>(x + i, y + j).val[2];

					median[(j + 1) * 3 + (i + 1)] = r + g + b;
				}
			}

			int temp[9];

			for (int i = 0; i < 9; i++)	temp[i] = median[i];

			for (int i = 0; i < 9; i++) {
				for (int j = i; j < 9; j++) {
					if (temp[i] > temp[j]) {
						int t = temp[i];
						temp[i] = temp[j];
						temp[j] = t;
					}
				}
			}

			//for (int i = 0; i < 9; i++) cout << temp[i] << " ";
			//cout << endl;

			for (int i = 0; i < 9; i++) {
				if (median[i] == temp[4]) {
					outputImage.at<Vec3b>(x, y) = inputImage.at<Vec3b>(x + i % 3, y + i / 3);
					break;
				}
			}
			*/
		}
	}

	//imshow("inputImage", inputImage);
	//imshow("outputImage", outputImage);
	//waitKey();

	return outputImage;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Pixel 값 덧셈
int pixelAdd(int pixel1, int pixel2)
{
	int temp = pixel1 + pixel2;
	//if (temp > 255) cout << "error" << endl;
	return (temp > 255) ? 255 : temp;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Pixel 값 뺄셈
int pixelSub(int pixel1, int pixel2)
{
	int temp = pixel1 - pixel2;
	return (temp < 0) ? 0 : temp;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Smoothing Filter 적용
Mat applyLinearFilter(Mat& ori, int* f, int row, int col)		// linear Filter 적용하는 함수
{
	Size oSize = ori.size();
	int fSize = row * col;

	Mat filterApplied(oSize, CV_8UC3);
	
	int fSum = 0;

	for (int i = 0; i < row; i++) {		// 필터의 각 요소의 합을 구함
		for (int j = 0; j < col; j++) {
			fSum += f[i * col + j];
		}
	}

	//cout << fSum << endl;

	if (fSum == 0) fSum = abs(f[(row - 1) / 2 * col + (col - 1) / 2]);		// Laplacian Filter 의 경우 합이 0 이므로 가운데 픽셀의 값을 차용

	for (int i = 0; i < oSize.width - col; i++) {			// 필터를 적용
		for (int j = 0; j < oSize.height - row; j++) {
			int sumB = 0;
			int sumG = 0;
			int sumR = 0;

			for (int x = 0; x < col; x++) {
				for (int y = 0; y < row; y++) {
					sumB += (int)ori.at<Vec3b>(Point(i + x, j + y)).val[0] * f[x + y * col];
					sumG += (int)ori.at<Vec3b>(Point(i + x, j + y)).val[1] * f[x + y * col];
					sumR += (int)ori.at<Vec3b>(Point(i + x, j + y)).val[2] * f[x + y * col];
				}
			}

			if(fSum != 0 && (sumB >= 0 && sumG >= 0 && sumR >= 0))
				filterApplied.at<Vec3b>(Point(i + (col - 1) / 2, j + (row - 1) / 2)) = Vec3b((float)sumB / fSum, (float)sumG / fSum, (float)sumR / fSum);
			if (fSum == 0 && (sumB >= 0 && sumG >= 0 && sumR >= 0))
				filterApplied.at<Vec3b>(Point(i + (col - 1) / 2, j + (row - 1) / 2)) = Vec3b(sumB, sumG, sumR);
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