#include <iostream>  
#include <cmath>
#include <iomanip>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;
using namespace std;


void Gaussian(float kernel[][3], float sigma)
{
	
	float sum = 0.0, temp;
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
	//for (int j = 0; j < response.rows; j++)
	
		//for (int i = 0; i < response.cols; i++)
	float kernel[3][3];

	for (int j = 1; j < image.rows - 1; j++)
		for (int i = 1; i < image.cols - 1; i++)
		{
			float sum = 0, sigma;
			//calculate sigma
			float sum_x = 0, sum_qua_x = 0;
			for (int y = 0; y < 3; y++)
				for (int x = 0; x < 3; x++)
				{
					sum_x += image.at<float>(j - 1 + y, i - 1 + x);
					sum_qua_x += pow(image.at<float>(j - 1 + y, i - 1 + x), 2);
				}
			float average = sum_x / 9;
			sigma = sqrt(sum_qua_x / 9 - pow(average, 2));
			cout << "simga:" << sigma << endl;
			Gaussian(kernel, sigma);

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

	Mat img = imread("house.PNG"), gray_img, Gaussian_img, img_ans;
	//namedWindow("123");
	//GaussianBlur(img, Gaussian_img, Size(3, 3), 0, 0, BORDER_DEFAULT);
	//cvtColor(Gaussian_img, gray_img, CV_RGB2GRAY);
	img_ans = img.clone();
	cvtColor(img, gray_img, CV_RGB2GRAY);

	Mat img_grad_x, img_grad_y, dst_img, norm_dst_img;
	Mat abs_grad_x, abs_grad_y;
	int patchSize = 2, apertureSize = 3;
	float k = 0.04f, thresh = 150;
	//(int)gray_img.at<float>(1, 1);
	cout << img.type();
	int scale = 1, delta = 0, ddepth = CV_32F;
	//cout << gray_img.at<Vec2b>(1, 1) << endl;
	Sobel(gray_img, img_grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	//convertScaleAbs(img_grad_x, abs_grad_x);
	Sobel(gray_img, img_grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	//convertScaleAbs(img_grad_y, abs_grad_y);
	//cout << abs_grad_x.at<float>(1, 1);


	cornerHarris(gray_img, dst_img, patchSize, apertureSize, k, BORDER_DEFAULT);
	//cout << dst_img.type();
	//imshow("dst_img", dst_img);
	normalize(dst_img, norm_dst_img, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	//imshow("norm_dst_img", norm_dst_img);
	//convertScaleAbs(norm_dst_img, dst_img);
	//cout << gray_img.at<double>(1, 1) << endl;
	//存取矩陣元素、存取圖片像素，行列次序剛好顛倒
	
	for (int j = 0; j < norm_dst_img.rows; j++)
	{
		for (int i = 0; i < norm_dst_img.cols; i++)
		{
			if ((int)norm_dst_img.at<float>(j, i) > thresh)
			{
				circle(img_ans, Point(i, j), 5, Scalar(0, 0, 255), 2, 8, 0);
			}
		}
	}
	imshow("Ans", img_ans);

	Mat img_grad_xx, img_grad_yy, img_grad_xy, det, trace, response, temp;
	
	multiply(img_grad_x, img_grad_x, img_grad_xx); //imshow("xx", img_grad_xx);
	multiply(img_grad_y, img_grad_y, img_grad_yy); //imshow("yy", img_grad_yy);
	multiply(img_grad_x, img_grad_y, img_grad_xy); //imshow("xy", img_grad_xy);
	
	//sum
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
	//boxFilter(img_grad_xx, img_grad_xx, img_grad_xx.depth(), Size(patchSize, patchSize),
	//	Point(-1, -1), false, BORDER_DEFAULT);
	//boxFilter(img_grad_yy, img_grad_yy, img_grad_yy.depth(), Size(patchSize, patchSize),
	//	Point(-1, -1), false, BORDER_DEFAULT);
	//boxFilter(img_grad_xy, img_grad_xy, img_grad_xy.depth(), Size(patchSize, patchSize),
	//	Point(-1, -1), false, BORDER_DEFAULT);
	normalize(img_grad_xx, img_grad_xx, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	normalize(img_grad_yy, img_grad_xx, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	normalize(img_grad_xy, img_grad_xx, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	Gaussian_filter(img_grad_xx);
	Gaussian_filter(img_grad_yy);
	Gaussian_filter(img_grad_xy);
	//GaussianBlur(img_grad_xx, img_grad_xx, Size(3, 3), 0, 0, BORDER_DEFAULT); //imshow("xx", img_grad_xx);
	//GaussianBlur(img_grad_yy, img_grad_yy, Size(3, 3), 0, 0, BORDER_DEFAULT); //imshow("yy", img_grad_yy);
	//GaussianBlur(img_grad_xy, img_grad_xy, Size(3, 3), 0, 0, BORDER_DEFAULT); //imshow("xy", img_grad_xy);


	//imshow("img_grad_xy", img_grad_xy);
	multiply(img_grad_xx, img_grad_yy, det);

	multiply(img_grad_xy, img_grad_xy, temp);
	det -= temp;

	trace = img_grad_xx + img_grad_yy;

	multiply(trace, trace, trace, k);
	//trace = k * trace;
	response = det - trace ;
	
	//Mat thresh = Mat(response.size(), response.type()), corner, ans;
	cout << response.type();
	//response.convertTo(ans, CV_32FC1);
	cout << (int)response.at<float>(2, 2);
	//thresh = Scalar(200);
	//compare(response, thresh, corner, CMP_GE);
	//imshow("response", response);
	//string s = response.type;
	normalize(response, response, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	//imshow("normalize", response);
	for (int j = 0; j < response.rows; j++)
	{
		for (int i = 0; i < response.cols; i++)
		{
			if ((int)response.at<float>(j, i) > thresh)
			{
				//cout << (int)response.at<unsigned char>(j, i) << "  " << endl;
				circle(img, Point(i, j), 5, Scalar(0, 0, 255), 2, 8, 0);
			}
			//cout << (int)response.at<float>(j, i);
		}
	}
	
	cout << img.size();
	imshow("456", img);
	//Mat grad;
	//addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
	//imshow("123", response);
	//imshow("456", abs_grad_y);
	//imshow("123", norm_dst_img);
	waitKey(60000);
}

