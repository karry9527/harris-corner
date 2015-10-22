/*
建置環境： visual studio 2013, opencv 3.0

*/

#include <iostream>  
#include <cmath>
#include <iomanip>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;
using namespace std;

//Gaussian kernel
void Gaussian(float kernel[][3])
{
	
	float sum = 0.0, temp, sigma = 1;
	for (int x = -1; x <= 1; x++)
		for (int y = -1; y <= 1; y++)
		{
			temp = exp(-(x*x + y*y) / (2 * pow(sigma, 2)));
			kernel[x + 1][y + 1] = temp / ((float)CV_PI * 2 * pow(sigma, 2));
			sum += kernel[x + 1][y + 1];
		}	
}

void Gaussian_filter(Mat &image)
{
	
	float kernel[3][3];
	Gaussian(kernel);
	for (int j = 1; j < image.rows - 1; j++)
		for (int i = 1; i < image.cols - 1; i++)
		{
			float sum = 0;
			for (int y = 0; y < 3; y++)
				for (int x = 0; x < 3; x++)
				{

					sum += image.at<float>(j - 1 + y, i - 1 + x) * kernel[x][y];
				}
			image.at<float>(j, i) = sum;
		}

}

int main()
{

	Mat img = imread("house.PNG"), gray_img, Gaussian_img;
	cvtColor(img, gray_img, CV_RGB2GRAY);

	Mat img_grad_x, img_grad_y;
	int patchSize = 2;
	float k = 0.04f, thresh;
	int scale = 1, delta = 0, ddepth = CV_32F;
	do{
		cout << "Please enter the thrshold(0~255)：";
		cin >> thresh;
	} while (!(thresh >= 0 && thresh <= 255));
	Sobel(gray_img, img_grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	Sobel(gray_img, img_grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);


	Mat img_grad_xx, img_grad_yy, img_grad_xy, det, trace, response, temp;
	
	multiply(img_grad_x, img_grad_x, img_grad_xx);
	multiply(img_grad_y, img_grad_y, img_grad_yy);
	multiply(img_grad_x, img_grad_y, img_grad_xy);
	
	//sum of Ixx, Iyy, Ixy
	for (int j = 0; j < img_grad_xx.rows - patchSize; j++)
		for (int i = 0; i < img_grad_xx.cols - patchSize; i++)
			for (int patch_y = j; patch_y < patchSize; patch_y++)
				for (int patch_x = i; patch_x < patchSize; patch_x++)
					if (!(patch_y == j && patch_x == i))
					{
						img_grad_xx.at<float>(j, i) += img_grad_xx.at<float>(patch_y, patch_x);
						img_grad_yy.at<float>(j, i) += img_grad_yy.at<float>(patch_y, patch_x);
						img_grad_xy.at<float>(j, i) += img_grad_xy.at<float>(patch_y, patch_x);

					}
	Gaussian_filter(img_grad_xx);
	Gaussian_filter(img_grad_yy);
	Gaussian_filter(img_grad_xy);

	//compute response
	multiply(img_grad_xx, img_grad_yy, det);
	multiply(img_grad_xy, img_grad_xy, temp);
	det -= temp;
	trace = img_grad_xx + img_grad_yy;
	multiply(trace, trace, trace, k);
	response = det - trace ;
	
	normalize(response, response, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

	//存取矩陣元素、存取圖片像素，行列次序剛好顛倒
	for (int j = 0; j < response.rows; j++)
	{
		for (int i = 0; i < response.cols; i++)
		{
			if ((int)response.at<float>(j, i) > thresh)
			{
				//circle the feature
				circle(img, Point(i, j), 5, Scalar(0, 0, 255), 2, 8, 0);
			}
		}
	}
	
	//output the pic
	imshow("harris corner", img);

	waitKey(60000);
}

