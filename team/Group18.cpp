// Group18.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#define _CRT_SECURE_NO_WARNINGS

#include <iostream>   
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp>   
#include <opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include<cmath>
#include <opencv2/opencv.hpp>
#include <string>

#define PI 3.14159265

using namespace std;  //省去屏幕输出函数cout前的std::
using namespace cv;   // 省去函数前面加cv::的必要性

void GaussFilter(Mat grayimg);//高斯平滑
int factorial(int n);//计算阶乘
Mat getSobelSmoooth(int wsize);
Mat getSobeldiff(int wsize);
void conv2D(Mat& src, Mat& dst, Mat kernel, int ddepth, Point anchor = Point(-1, -1), int delta = 0, int borderType = BORDER_DEFAULT);
void sepConv2D_Y_X(cv::Mat& src, Mat& dst, Mat kernel_Y, Mat kernel_X, int ddepth, Point anchor = Point(-1, -1), int delta = 0, int borderType = BORDER_DEFAULT);
void sepConv2D_X_Y(Mat& src, Mat& dst, Mat kernel_X, Mat kernel_Y, int ddepth, Point anchor = Point(-1, -1), int delta = 0, int borderType = BORDER_DEFAULT);
void Sobel(Mat& src, Mat& dst_X, Mat& dst_Y, Mat& dst, int wsize, int ddepth, Point anchor = Point(-1, -1), int delta = 0, int borderType = cv::BORDER_DEFAULT);
bool checkInRang(int r, int c, int rows, int cols);
void trace(Mat& edgeMag_noMaxsup, Mat& edge, float TL, int r, int c, int rows, int cols);
void Edge_Canny(Mat& src, Mat& edge, float TL, float TH, int wsize = 3, bool L2graydient = false);

void hough_lines(Mat& img, int threshold, vector<Vec2f> lines);

//高斯平滑
void GaussFilter(Mat grayimg)
{
	unsigned char* pimg = grayimg.data;//图像地址
	int height = grayimg.rows;
	int width = grayimg.cols;
	//为新图像处理部分分配存储空间（不包括边界点
	Mat imgnew;
	imgnew.create(height, width, CV_8UC1);
	unsigned char* pnewimg = imgnew.data;
	for (int i = 1; i < height - 1; i++)
	{
		for (int j = 1; j < width - 1; j++)
		{
			pnewimg[i * width + j] = 0;
		}
	}//初始化

	 //读取保存在txt中的滤波算子
	FILE* fp;
	int n = 0;
	fp = fopen("gausslvbo.txt", "r");

	if (!fp)
	{
		printf("无法读取滤波算子\n");
	}

	double H[9];//存储算子
	for (int i = 0; i < 9; i++)
	{
		fscanf_s(fp, "%lf", &H[i]);
	}

	//卷积运算
	for (int i = 1; i < height - 1; i++)
	{ //从1开始，先不处理边界点
		for (int j = 1; j < width - 1; j++)
		{
			double t = 0;
			for (int m = 0; m < 3; m++)
			{//选取所选点8-领域的点
				for (int n = 0; n < 3; n++)
				{
					t += (double)pimg[(i + m - 1) * width + j + n - 1] * H[m * 3 + n];
				}
			}
			pnewimg[i * width + j] = (unsigned char)abs(t);
			if (pnewimg[i * width + j] > 255)
			{
				pnewimg[i * width + j] = 255;
			}
		}
	}
	//再对边界点进行处理(保留
	int t = 0;//第一行
	for (int q = 0; q < width; q++)
	{
		pnewimg[t * width + q] = pimg[t * width + q];
	}
	t = height - 1;//最后一行
	for (int q = 0; q < width; q++)
	{
		pnewimg[t * width + q] = pimg[t * width + q];
	}
	int q = 0;//第一列
	for (int t = 0; t < height; t++)
	{
		pnewimg[t * width + q] = pimg[t * width + q];
	}
	q = width - 1;//最后一列
	for (int t = 0; t < height; t++)
	{
		pnewimg[t * width + q] = pimg[t * width + q];
	}

	namedWindow("高斯平滑", WINDOW_NORMAL);
	imshow("高斯平滑", imgnew); //显示变换后的图像
	imwrite("smoothimg.bmp", imgnew);//保存图片
	waitKey();
}


/**********************Sobel算子*************************/
//阶乘
int factorial(int n) 
{
	int fac = 1;
	//0的阶乘
	if (n == 0)
		return fac;
	for (int i = 1; i <= n; ++i) 
    {
		fac *= i;
	}
	return fac;
}

//获得Sobel平滑算子
Mat getSobelSmoooth(int wsize) 
{
	int n = wsize - 1;
	Mat SobelSmooothoper = Mat::zeros(Size(wsize, 1), CV_32FC1);
	for (int k = 0; k <= n; k++) 
    {
		float* pt = SobelSmooothoper.ptr<float>(0);//指向第一行第一个元素
		pt[k] = factorial(n) / (factorial(k) * factorial(n - k));
	}
	return SobelSmooothoper;
}

//获得Sobel差分算子
Mat getSobeldiff(int wsize) 
{
	Mat Sobeldiffoper = Mat::zeros(Size(wsize, 1), CV_32FC1);
	Mat SobelSmoooth = getSobelSmoooth(wsize - 1);
	for (int k = 0; k < wsize; k++) {
		if (k == 0)
			Sobeldiffoper.at<float>(0, k) = 1;
		else if (k == wsize - 1)
			Sobeldiffoper.at<float>(0, k) = -1;
		else
			Sobeldiffoper.at<float>(0, k) = SobelSmoooth.at<float>(0, k) - SobelSmoooth.at<float>(0, k - 1);
	}
	return Sobeldiffoper;
}

//可分离卷积———先垂直方向卷积，后水平方向卷积
void sepConv2D_Y_X(Mat& src, Mat& dst, Mat kernel_Y, Mat kernel_X, int ddepth, Point anchor , int delta, int borderType ) 
{
	Mat dst_kernel_Y;
    filter2D(src, dst_kernel_Y, ddepth, kernel_Y, anchor, delta, borderType); //垂直方向卷积
    filter2D(dst_kernel_Y, dst, ddepth, kernel_X, anchor, delta, borderType); //水平方向卷积
}

//可分离卷积———先水平方向卷积，后垂直方向卷积
void sepConv2D_X_Y(Mat& src, Mat& dst, Mat kernel_X, Mat kernel_Y, int ddepth, Point anchor , int delta , int borderType ) 
{
	Mat dst_kernel_X;
    filter2D(src, dst_kernel_X, ddepth, kernel_X, anchor, delta, borderType); //水平方向卷积
    filter2D(dst_kernel_X, dst, ddepth, kernel_Y, anchor, delta, borderType); //垂直方向卷积
}

/*************************************************************************************************/

//Sobel算子边缘检测
//dst_X 垂直方向
//dst_Y 水平方向
void Sobel(Mat& src, Mat& dst_X, Mat& dst_Y, Mat& dst, int wsize, int ddepth, Point anchor , int delta, int borderType )
{
	Mat SobelSmooothoper = getSobelSmoooth(wsize); //平滑系数
	Mat Sobeldiffoper = getSobeldiff(wsize); //差分系数

    //可分离卷积———先垂直方向平滑，后水平方向差分——得到垂直边缘
	sepConv2D_Y_X(src, dst_X, SobelSmooothoper.t(), Sobeldiffoper, ddepth);

	//可分离卷积———先水平方向平滑，后垂直方向差分——得到水平边缘
	sepConv2D_X_Y(src, dst_Y, SobelSmooothoper, Sobeldiffoper.t(), ddepth);

	//边缘强度（近似）
	dst = abs(dst_X) + abs(dst_Y);
	convertScaleAbs(dst, dst); //求绝对值并转为无符号8位图
}

//确定一个点的坐标是否在图像内
bool checkInRang(int r, int c, int rows, int cols) 
{
	if (r >= 0 && r < rows && c >= 0 && c < cols)
		return true;
	else
		return false;
}

//从确定边缘点出发，延长边缘
void trace(Mat& edgeMag_noMaxsup, Mat& edge, float TL, int r, int c, int rows, int cols) 
{
	if (edge.at<uchar>(r, c) == 0) 
    {
		edge.at<uchar>(r, c) = 255;
		for (int i = -1; i <= 1; ++i) //继续判断该边缘点3*3邻域内是否有大于高阈值的点
        {
			for (int j = -1; j <= 1; ++j) 
            {
				float mag = edgeMag_noMaxsup.at<float>(r + i, c + j);
				if (checkInRang(r + i, c + j, rows, cols) && mag >= TL)
					trace(edgeMag_noMaxsup, edge, TL, r + i , c + j , rows, cols);
			}
		}
	}
}

//Canny边缘检测
void Edge_Canny(Mat& src, Mat& edge, float TL, float TH, int wsize , bool L2graydient ) 
{

	int rows = src.rows;
	int cols = src.cols;

	//sobel算子
	Mat dx, dy, sobel_dst;
	Sobel(src, dx, dy, sobel_dst, wsize, CV_32FC1);

	//计算梯度幅值
	Mat edgeMag;
	if (L2graydient)
		magnitude(dx, dy, edgeMag); //开平方
	else
		edgeMag = abs(dx) + abs(dy); //绝对值之和近似

	edgeMag.convertTo(edgeMag, CV_8UC1, 1.0, 0);//将结果转换 CV_8U 类型
	namedWindow("梯度幅值", WINDOW_NORMAL);
	imshow("梯度幅值", edgeMag); //显示变换后的图像
	waitKey();

	//计算梯度方向 以及 非极大值抑制
	edgeMag.convertTo(edgeMag, CV_32FC1, 1.0, 0);
	Mat edgeMag_noMaxsup = Mat::zeros(rows, cols, CV_32FC1);
	for (int r = 1; r < rows - 1; ++r) 
    {
		for (int c = 1; c < cols - 1; ++c) 
        {
			float x = dx.at<float>(r, c);
			float y = dy.at<float>(r, c);
			float angle = std::atan2f(y, x) / CV_PI * 180; //当前位置梯度方向
			float mag = edgeMag.at<float>(r, c);  //当前位置梯度幅值

	        //非极大值抑制
			//梯度方向为水平方向-3*3邻域内左右方向比较
			if (abs(angle) < 22.5 || abs(angle) > 157.5)
            {
				float left = edgeMag.at<float>(r, c - 1);
				float right = edgeMag.at<float>(r, c + 1);
				if (mag >= left && mag >= right)
					edgeMag_noMaxsup.at<float>(r, c) = mag;
			}

			//梯度方向为垂直方向-3*3邻域内上下方向比较
			if ((angle >= 67.5 && angle <= 112.5) || (angle >= -112.5 && angle <= -67.5))
            {
				float top = edgeMag.at<float>(r - 1, c);
				float down = edgeMag.at<float>(r + 1, c);
				if (mag >= top && mag >= down)
					edgeMag_noMaxsup.at<float>(r, c) = mag;
			}

			//梯度方向为-45°-3*3邻域内右上左下方向比较
			if ((angle > 112.5 && angle <= 157.5) || (angle > -67.5 && angle <= -22.5))
            {
				float right_top = edgeMag.at<float>(r - 1, c + 1);
				float left_down = edgeMag.at<float>(r + 1, c - 1);
				if (mag >= right_top && mag >= left_down)
					edgeMag_noMaxsup.at<float>(r, c) = mag;
			}

			//梯度方向为+45°-3*3邻域内右下左上方向比较
			if ((angle >= 22.5 && angle < 67.5) || (angle >= -157.5 && angle < -112.5))
            {
				float left_top = edgeMag.at<float>(r - 1, c - 1);
				float right_down = edgeMag.at<float>(r + 1, c + 1);
				if (mag >= left_top && mag >= right_down)
					edgeMag_noMaxsup.at<float>(r, c) = mag;
			}
		}
	}
	edgeMag_noMaxsup.convertTo(edgeMag_noMaxsup, CV_8UC1, 1.0, 0);//将结果转换 CV_8U 类型
	namedWindow("非极大值抑制", WINDOW_NORMAL);
	imshow("非极大值抑制", edgeMag_noMaxsup); //显示变换后的图像
	waitKey();

	//双阈值处理及边缘连接
	edgeMag_noMaxsup.convertTo(edgeMag_noMaxsup, CV_32FC1, 1.0, 0);
	edge = Mat::zeros(rows, cols, CV_8UC1);
	for (int r = 1; r < rows - 1; ++r) 
    {
		for (int c = 1; c < cols - 1; ++c)  
        {
			float mag = edgeMag_noMaxsup.at<float>(r, c);
			//大于高阈值，确定为边缘点
			if (mag >= TH)
				trace(edgeMag_noMaxsup, edge, TL, r, c, rows, cols);
            //小于低阈值，排除为边缘点
			else if (mag < TL)
				edge.at<uchar>(r, c) = 0;
		}
	}

}


/***************************Hough变换****************************/
void hough_lines(Mat& img, Mat& image_output, float rho, float theta, int threshold) 
{
	AutoBuffer<int> _accum, _sort_buf;
	AutoBuffer<float> _tabSin, _tabCos;

	const uchar* image;
	int step, width, height;
	int numangle, numrho;
	int total = 0;
	int i, j;
	float irho = 1 / rho;
	double scale;

	image = img.ptr();    //得到图像的指针
	step = img.step;    //得到图像的步长
	width = img.cols;    //得到图像的宽
	height = img.rows;    //得到图像的高
	//由角度和距离的分辨率得到角度和距离的数量，即霍夫变换后角度和距离的个数
	numangle = cvRound(CV_PI / theta);
	numrho = cvRound(((width + height) * 2 + 1) / rho);

	_accum.allocate((numangle + 2) * (numrho + 2));
	//为排序数组分配内存空间
	_sort_buf.allocate(numangle * numrho);
	//为正弦和余弦列表分配内存空间
	_tabSin.allocate(numangle);
	_tabCos.allocate(numangle);
	//分别定义上述内存空间的地址指针
	int* accum = _accum, * sort_buf = _sort_buf;
	float* tabSin = _tabSin, * tabCos = _tabCos;
	//累加器数组清零
	memset(accum, 0, sizeof(accum[0]) * (numangle + 2) * (numrho + 2));

	float ang = 0;
	//为避免重复运算，事先计算好sinθi/ρ和cosθi/ρ
	for (int n = 0; n < numangle; ang += theta, n++)
	{
		tabSin[n] = (float)(sin((double)ang) * irho);
		tabCos[n] = (float)(cos((double)ang) * irho);
	}

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			//只对图像的非零值处理，即只对图像的边缘像素进行霍夫变换
			if (image[i * step + j] != 0)
				for (int n = 0; n < numangle; n++)
				{
					int r = cvRound(j * tabCos[n] + i * tabSin[n]);
					r += (numrho - 1) / 2;
					//r表示的是距离，n表示的是角点，在累加器内找到它们所对应的位置（即霍夫空间内的位置），其值加1
					accum[(n + 1) * (numrho + 2) + r + 1]++;
				}
		}

	for (int r = 0; r < numrho; r++)
		for (int n = 0; n < numangle; n++)
		{
			//得到当前值在累加器数组的位置
			int base = (n + 1) * (numrho + 2) + r + 1;
			if (accum[base] > threshold &&    //必须大于所设置的阈值
				//在4邻域内进行非极大值抑制
				accum[base] > accum[base - 1] && accum[base] >= accum[base + 1] &&
				accum[base] > accum[base - numrho - 2] && accum[base] >= accum[base + numrho + 2])
				//把极大值位置存入排序数组内——sort_buf
				sort_buf[total++] = base;
		}

	//事先定义一个尺度
	scale = 1. / (numrho + 2);
	vector<Vec2f> lines;
	for (i = 0; i < total; i++)
	{
		Vec2f temp;
		//idx为极大值在累加器数组的位置
		int idx = sort_buf[i];
		//分离出该极大值在霍夫空间中的位置
		int n = cvFloor(idx * scale) - 1;
		int r = idx - (n + 1) * (numrho + 2) - 1;
		//最终得到极大值所对应的角度和距离
		temp[0] = (r - (numrho - 1) * 0.5f) * rho;
		temp[1] = n * theta;
		//存储到序列内
		cout << temp[0] << endl;
		cout << temp[1] << endl;
		lines.push_back(temp);
	}


	vector<Vec2f>::const_iterator it = lines.begin();
	while (it != lines.end())
	{
		float rho = (*it)[0];
		float theta = (*it)[1];
		if (theta < PI / 4. || theta > 3. * PI / 4.)
		{
			Point pt1(rho / cos(theta), 0);
			Point pt2((rho - image_output.rows * sin(theta)) / cos(theta), image_output.rows);
			line(image_output, pt1, pt2, Scalar(0, 0, 255), 1);
		}
		else
		{
			Point pt1(0, rho / sin(theta));
			Point pt2(image_output.cols, (rho - image_output.cols * cos(theta)) / sin(theta));
			line(image_output, pt1, pt2, Scalar(0, 0, 255), 1);
		}
		++it;
	}
};

class LineFinder 
{
private:
	vector<Vec2f> lines;
	double delta_rho;
	double delta_theta;
	int threshold;
public:
	LineFinder() {
		delta_rho = 1;
		delta_theta = PI / 180;
		threshold = 80;
	}
	void setAccResolution(double dRho, double dTheta) {
		delta_rho = dRho;
		delta_theta = dTheta;
	}
	void setthreshold(int minv) {
		threshold = minv;
	}
	void findLines(Mat& binary, Mat& image_output) {
		lines.clear();
		hough_lines(binary, image_output, delta_rho, delta_theta, threshold);
		//HoughLines(binary, lines, delta_rho, delta_theta, threshold);
		for (int i = 0; i < lines.size(); i++) {
			cout << lines[i][0] << endl;
		}
		cout << lines.size();
	}

};

int main()
{ 
    Mat image_input = imread("20180620-tianjin.bmp", IMREAD_ANYCOLOR);   // 读入图片 

    if (image_input.empty())     // 判断文件是否正常打开  
    {
        fprintf(stderr, "Can not load image %s\n", "20180620-tianjin.bmp");
        waitKey(6000);  // 等待6000 ms后窗口自动关闭   
        return -1;
    }
	namedWindow("原图", WINDOW_NORMAL);
	imshow("原图", image_input); //显示原图

	Mat image_gray;
	cvtColor(image_input, image_gray, COLOR_RGB2GRAY);
	GaussFilter(image_gray);//高斯平滑

    Mat gray = imread("smoothimg.bmp", IMREAD_GRAYSCALE);
    Edge_Canny(gray, image_gray, 80, 160, 3, 1 );//进行一次canny边缘检测

    namedWindow("Edge", WINDOW_NORMAL);
    imshow("Edge", image_gray); //显示Canny边缘检测后的图像
  	Mat image_output(image_gray.rows, image_gray.cols, CV_8U, Scalar(255));
	image_input.copyTo(image_output);
	LineFinder finder;
	finder.setthreshold(130);//Hough变换
	finder.findLines(image_gray, image_output);

	namedWindow("Hough", WINDOW_NORMAL);
	imshow("Hough", image_output);
	waitKey(0);
	return 0;

}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
