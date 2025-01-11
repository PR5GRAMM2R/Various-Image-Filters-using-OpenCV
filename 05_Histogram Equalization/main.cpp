// PSNR.cpp : �ܼ� ���� ���α׷��� ���� �������� �����մϴ�.
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

	cout << "���� �� : ";
	cin >> x;*/

	Mat img_in;

	// image �а� gray�� �ٲٱ�
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
	// Step 1 ó���ϱ�
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	Mat lena1(output_y_size, output_x_size, CV_8UC1);			// lena1 ����

	unsigned char *pData1;
	pData1 = new unsigned char[output_x_size * output_y_size];
	
	///////////////////// ó���ϱ� ///////////////////
	for (int i = 0; i < output_x_size * output_y_size; i++)
	{
		pData1[i] = (unsigned char)((double)(original.data[i]) / 2.0);				// s = r / 2 �� ��ȯ�� ���� lena1 �����

		pData1[i] = pData1[i] > 255 ? 255 : pData1[i];
		pData1[i] = pData1[i] < 0 ? 0 : pData1[i];
	}

	lena1.data = (uchar*)pData1;
	imshow("lena1 image", lena1);

	Mat lena2(output_y_size, output_x_size, CV_8UC1);			// lena2 ����

	unsigned char *pData2;
	pData2 = new unsigned char[output_x_size * output_y_size];
	
	///////////////////// ó���ϱ� ///////////////////
	for (int i = 0; i < output_x_size * output_y_size; i++)
	{
		pData2[i] = (unsigned char)(128.0 + (double)(original.data[i]) / 2.0);		// s = 128 + r / 2 �� ��ȯ�� ���� lena2 �����

		pData2[i] = pData2[i] > 255 ? 255 : pData2[i];
		pData2[i] = pData2[i] < 0 ? 0 : pData2[i];
	}

	lena2.data = (uchar*)pData2;
	imshow("lena2 image", lena2);

	//waitKey(0);

	//return 0;

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Step 2 ó���ϱ�
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