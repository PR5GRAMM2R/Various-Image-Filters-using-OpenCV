// PSNR.cpp : �ܼ� ���� ���α׷��� ���� �������� �����մϴ�.
//

#include "stdafx.h"			// 2020203011 ����ȯ

#define _USE_MATH_DEFINES
#include <math.h>

//#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
//#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;

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

	cout << "���� �� : ";
	cin >> x;*/

	Mat img_in;

	// image �а� gray�� �ٲٱ�
	img_in = imread("Firework.jpg");

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
	// Moving Average ( 3 x 3  ũ�� ) �����ϱ�
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	int movingAverageFilter[3 * 3]  = {	1, 1, 1,				// Moving Average Filter �� ���� 3 x 3 ����
									1, 1, 1,
									1, 1, 1 };

	Mat movingAverageFilterApplied(applyLinearFilter(original, movingAverageFilter, 3, 3));	// ���� �����ϱ�

	imshow("movingAverageFilterApplied", movingAverageFilterApplied);
	//waitKey();


	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Laplacian ( 3 x 3  ũ�� ) �����ϱ�
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	int laplacianFilter[3 * 3]  = {	1, 1, 1,					// Laplacian ����
									1, -8, 1,
									1, 1, 1 };

	Mat LaplacianFilterApplied(applyLinearFilter(original, laplacianFilter, 3, 3));	// ���� �����ϱ�

	imshow("LaplacianFilterApplied", LaplacianFilterApplied);
	//waitKey();

	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Sharpening Filter �����ϱ�
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	int sharpeningFilter[3 * 3] = { 0, 1, 0,					// Sharpening �� ���� Laplacian Filter
									1, -4, 1,
									0, 1, 0 };

	Mat sharpeningFilterApplied(applyLinearFilter(original, sharpeningFilter, 3, 3));	// ���� �����ϱ�
	
	Mat sharpeningFilterResult(size, CV_8UC3);

	for (int i = 0; i < size.width; i++) {
		for (int j = 0; j < size.height; j++) {
			int b = original.at<Vec3b>(Point(i, j)).val[0] + sharpeningFilterApplied.at<Vec3b>(Point(i, j)).val[0];		// ���� �̹����� Sharpening Filter �ռ��ϱ�
			int g = original.at<Vec3b>(Point(i, j)).val[1] + sharpeningFilterApplied.at<Vec3b>(Point(i, j)).val[1];
			int r = original.at<Vec3b>(Point(i, j)).val[2] + sharpeningFilterApplied.at<Vec3b>(Point(i, j)).val[2];

			b = (b > 255) ? 255 : b;		// ������ ����� �� �����ϱ�
			g = (g > 255) ? 255 : g;
			r = (r > 255) ? 255 : r;

			b = (b < 0) ? 0 : b;
			g = (g < 0) ? 0 : g;
			r = (r < 0) ? 0 : r;

			sharpeningFilterResult.at<Vec3b>(Point(i, j)) = Vec3b(b, g, r);
		}
	}

	imshow("sharpeningFilterResult", sharpeningFilterResult);
	waitKey();

	
	return 0;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Smoothing Filter ����
Mat applyLinearFilter(Mat& ori, int* f, int row, int col)		// linear Filter �����ϴ� �Լ�
{
	Size oSize = ori.size();
	int fSize = row * col;

	Mat filterApplied(oSize, CV_8UC3);
	
	int fSum = 0;

	for (int i = 0; i < row; i++) {		// ������ �� ����� ���� ����
		for (int j = 0; j < col; j++) {
			fSum += f[i * col + j];
		}
	}

	//cout << fSum << endl;

	if (fSum == 0) fSum = abs(f[(row - 1) / 2 * col + (col - 1) / 2]);		// Laplacian Filter �� ��� ���� 0 �̹Ƿ� ��� �ȼ��� ���� ����

	for (int i = 0; i < oSize.width - col; i++) {			// ���͸� ����
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
// Histogram Equalization ����
Mat getHistogramEqualization(Mat& input)
{
	int output_x_size = input.cols;
	int output_y_size = input.rows;
	
	int L = 256;		// 8 bit �̹Ƿ� L �� 256

	int n[256] = { 0 };			// �� �ȼ��� 256 ������ ���� �� �ϳ��� ���� �� �����Ƿ�, �� �������� �ȼ� ���� �迭�μ� ����

	for (int i = 0; i < output_x_size * output_y_size; i++) {
		n[input.data[i]]++;
	}

	double p[256] = { 0.0 };		// ��ü �ȼ� ������ �������� �� �������� �ȼ� ���� ������ ����

	for (int i = 0; i < L; i++) {
		p[i] = (double)(n[i]) / (double)(output_x_size * output_y_size);
	}

	int s[256] = { 0 };				// ó�� ���� �ȼ����� Histogram Equalization �� ���� � �ȼ������� �ٲ�� �� ����

	for (int i = 0; i < L; i++) {
		double temp = 0;

		for (int j = 0; j < i + 1; j++) {
			temp += p[j];
		}

		s[i] = round((double)(L - 1) * temp);		// ������� �ݿø��Ͽ� s �迭�� ����
	}

	Mat output(output_y_size, output_x_size, CV_8UC1);
	
	unsigned char* pData = new unsigned char[output_x_size * output_y_size];

	for (int i = 0; i < output_x_size * output_y_size; i++) {
		pData[i] = s[input.data[i]];			// s �迭�� �����Ͽ� �� �ȼ������� ��ȯ����
	}

	output.data = pData;

	return output;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Negative Transformation  ����
unsigned char getNegativeTransformation(unsigned char inputPixel)
{
	return 255 - inputPixel;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Gamma Transformation ����
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
// �ȼ��� �����̵�
Point transposition(Point start, Point dest)
{
	return Point(start.x + dest.x, start.y + dest.y);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// �ȼ��� ȸ���̵�
Point rotation(Point start, double rot)
{
	double x = cos(rot) * start.x - sin(rot) * start.y;
	double y = sin(rot) * start.x + cos(rot) * start.y;

	return Point(x, y);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Bilinear Interpolation ����
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

	double colPosTemp = pixel.x;	// x ��		// output �� �ȼ� ��ġ ����
	double rowPosTemp = pixel.y;	// y ��		// output �� �ȼ� ��ġ ����

	double colPos = colPosTemp * outputDeltaX;	// x ��		// input �� ���� output �� �ȼ� ��ġ ����
	double rowPos = rowPosTemp * outputDeltaY;	// y ��

	colPos = (colPos < 0) ? 0 : colPos;		// ���� �ȼ� ������ ����� ���� ����
	colPos = (colPos > 1) ? 1 : colPos;
	rowPos = (rowPos < 0) ? 0 : rowPos;
	rowPos = (rowPos > 1) ? 1 : rowPos;

	int upLeftColPos = (colPos == 1) ? img_x_size - 1 : (int)trunc(colPos * (img_x_size - 1));	// output �ȼ� �ֺ��� �簢���� �»�� �ȼ� ��ǥ�� ����
	int upLeftRowPos = (rowPos == 1) ? img_y_size - 1 : (int)trunc(rowPos * (img_y_size - 1));
	unsigned int upLeftValue = inputImage.data[upLeftRowPos * img_x_size + upLeftColPos];

	int upRightColPos = upLeftColPos + 1;																					// output �ȼ� �ֺ��� �簢���� ���� �ȼ� ��ǥ�� ����
	int upRightRowPos = upLeftRowPos;
	unsigned int upRightValue = inputImage.data[upRightRowPos * img_x_size + upRightColPos];

	int downLeftColPos = upLeftColPos;																						// output �ȼ� �ֺ��� �簢���� ���ϴ� �ȼ� ��ǥ�� ����
	int downLeftRowPos = upLeftRowPos + 1;
	unsigned int downLeftValue = inputImage.data[downLeftRowPos * img_x_size + downLeftColPos];

	int downRightColPos = upLeftColPos + 1;																					// output �ȼ� �ֺ��� �簢���� ���ϴ� �ȼ� ��ǥ�� ����
	int downRIghtRowPos = upLeftRowPos + 1;
	unsigned int downRIghtValue = inputImage.data[downRIghtRowPos * img_x_size + downRightColPos];

	double dx = (colPos - ((double)upLeftColPos * inputDeltaX)) * (img_x_size - 1);
	double dy = (rowPos - ((double)upLeftRowPos * inputDeltaY)) * (img_y_size - 1);

	double sum = (double)upLeftValue * (1 - dx) * (1 - dy) + (double)upRightValue * dx * (1 - dy) + (double)downLeftValue * (1 - dx) * dy + (double)downRIghtValue * dx * dy;

	return (unsigned char)round(sum);// / (imgDeltaX * imgDeltaY));
}