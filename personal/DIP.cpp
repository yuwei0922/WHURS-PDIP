// DIP.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>   
#include <opencv2\imgproc\imgproc.hpp>  
#include <cstdlib>

using namespace std;  //省去屏幕输出函数cout前的std::
using namespace cv;   // 省去函数前面加cv::的必要性

//添加盐燥声
void salt(Mat image, int n)
{
	int i, j;
	for (int k = 0; k < n; k++)
	{
		// rand()是随机数生成器
		i = rand() % image.cols;
		j = rand() % image.rows;
		if (image.type() == CV_8UC1)
		{ // 灰度图像
			image.at<uchar>(j, i) = 255;
		}
		else if (image.type() == CV_8UC3)
		{ // 彩色图像
			image.at<cv::Vec3b>(j, i)[0] = 255;
			image.at<cv::Vec3b>(j, i)[1] = 255;
			image.at<cv::Vec3b>(j, i)[2] = 255;
		}
	}
}

//添加椒噪声
void pepper(Mat image, int n)
{
	int i, j;
	for (int k = 0; k < n; k++)
	{
		// rand()是随机数生成器
		i = rand() % image.cols;
		j = rand() % image.rows;
		if (image.type() == CV_8UC1)
		{ // 灰度图像
			image.at<uchar>(j, i) = 0;
		}
		else if (image.type() == CV_8UC3)
		{ // 彩色图像
			image.at<cv::Vec3b>(j, i)[0] = 0;
			image.at<cv::Vec3b>(j, i)[1] = 0;
			image.at<cv::Vec3b>(j, i)[2] = 0;
		}
	}
}

int radius = 20;//截断频率
int lpType = 0;//低通滤波器的类型
const int Max_RADIUS = 100;//设置最大的截断频率
const int MAX_LPTYPE = 2;//设置滤波类型

Mat I;//低通&高通滤波局部处理
Mat F;//图像的快速傅里叶变换
Mat FlpSpectrum;//低通傅里叶变换的傅里叶谱灰度级
Mat FhpSpectrum;//高通傅里叶变换的傅里叶谱灰度级
Mat F_lpFilter;//低通傅里叶变换
Mat F_hpFilter;//高通傅里叶变换

Point maxLoc;	//Point maxLoc为傅里叶谱的最大值的坐标

Mat Img_Rotate; //图像旋转
int degree = 0;//旋转角度
int scale = 1;//缩放倍数
const int MAX_DEGREE = 360; //设置最大的旋转角度
const int MAX_SCALE = 5; //设置最大的缩放倍数

Mat Img_Translation; //图像旋转
int dx = 0;//x方向
int dy = 0;//y方向
const int MAX_DX = 500; //设置x方向最大的平移距离
const int MAX_DY = 500; //设置y方向最大的平移距离

//灰度线性变换
void LinearProc(Mat src)
{
	namedWindow("原图", CV_WINDOW_AUTOSIZE);
	imshow("原图", src);
	Mat dst;
	int rows = src.rows;
	int cols = src.cols;
	float alpha = 1.2, beta = 10;
	dst = Mat::zeros(src.size(), src.type());
	for (int row = 0; row < rows; row++) 
	{
		for (int col = 0; col < cols; col++) 
		{
			if (src.channels() == 3) 
			{
				//彩色图像
				int b = src.at<Vec3b>(row, col)[0];//获取三个通道灰度值
				int g = src.at<Vec3b>(row, col)[1];
				int r = src.at<Vec3b>(row, col)[2];
				dst.at<Vec3b>(row, col)[0] = saturate_cast<uchar>((alpha * b + beta));//线性变换
				dst.at<Vec3b>(row, col)[1] = saturate_cast<uchar>((alpha * g + beta));
				dst.at<Vec3b>(row, col)[2] = saturate_cast<uchar>((alpha * r + beta));
			}
			else if (src.channels() == 1) 
			{
				//灰度图像
				int v = src.at<uchar>(row, col);
				dst.at<uchar>(row, col) = saturate_cast<uchar>(alpha * v + beta);
			}
		}
	}

	imshow("灰度线性变换结果", dst);
	waitKey(0);
}

//快速傅里叶变换
void fft2Image(InputArray _src, OutputArray _dst)
{
	//将InputArray数据类型转换成Mat类型
	Mat src = _src.getMat();
	//判断位深和通道
	CV_Assert(src.type() == CV_32FC1 || src.type() == CV_64FC1);
	CV_Assert(src.channels() == 1 || src.channels() == 2);
	int rows = src.rows;
	int cols = src.cols;
	//为了进行快速的傅里叶变换，经行和列的扩充,找到最合适扩充值
	Mat padded;
	int rPadded = getOptimalDFTSize(rows);
	int cPadded = getOptimalDFTSize(cols);
	//进行边缘扩充,扩充值为零
	copyMakeBorder(src, padded, 0, rPadded - rows, 0, cPadded - cols, BORDER_CONSTANT, Scalar::all(0));
	//快速的傅里叶变换（双通道：用于存储实部 和 虚部）
	dft(padded, _dst, DFT_COMPLEX_OUTPUT);
}

//幅度谱
void amplitudeSpectrum(InputArray _srcFFT, OutputArray _dstSpectrum)
{
	//判断傅里叶变换是两个通道（实部和虚部）
	CV_Assert(_srcFFT.channels() == 2);
	//分离通道
	vector<Mat> FFT2Channel;
	split(_srcFFT, FFT2Channel);
	//计算傅里叶变换的幅度谱 sqrt(pow(R,2)+pow(I,2))
	magnitude(FFT2Channel[0], FFT2Channel[1], _dstSpectrum);
}

//幅度谱的灰度级显示
Mat graySpectrum(Mat spectrum)
{
	Mat dst;
	log(spectrum + 1, dst);
	//归一化
	normalize(dst, dst, 0, 1, NORM_MINMAX);
	//为了进行灰度级显示，做类型转换
	dst.convertTo(dst, CV_8UC1, 255, 0);
	return dst;
}

//构造低通滤波器
enum LPFILTER_TYPE { ILP_FILTER = 0, BLP_FILTER = 1, GLP_FILTER = 2 };
Mat createLPFilter(Size size, Point center, float radius, int type, int n = 2)
{
	Mat lpFilter = Mat::zeros(size, CV_32FC1);
	int rows = size.height;
	int cols = size.width;
	if (radius <= 0)
		return lpFilter;
	//构造理想低通滤波器
	if (type == ILP_FILTER)
	{
		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				float norm2 = pow(abs(float(r - center.y)), 2) + pow(abs(float(c - center.x)), 2);
				if (sqrt(norm2) < radius)
					lpFilter.at<float>(r, c) = 1;
				else
					lpFilter.at<float>(r, c) = 0;
			}
		}
	}
	//构造二阶巴特沃斯低通滤波器
	if (type == BLP_FILTER)
	{
		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				lpFilter.at<float>(r, c) = float(1.0 / (1.0 + pow(sqrt(pow(r - center.y, 2.0) + pow(c - center.x, 2.0)) / radius, 2.0 * n)));
			}
		}
	}
	//构造高斯低通滤波
	if (type == GLP_FILTER)
	{
		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				lpFilter.at<float>(r, c) = float(exp(-(pow(c - center.x, 2.0) + pow(r - center.y, 2.0)) / (2 * pow(radius, 2.0))));
			}
		}
	}
	return lpFilter;
}

//构造高通滤波器
Mat createHPFilter(Size size, Point center, float radius, int type, int n = 2)
{
	Mat hpFilter = Mat::zeros(size, CV_32FC1);
	int rows = size.height;
	int cols = size.width;
	if (radius <= 0)
		return hpFilter;

	//构造理想高通滤波器
	if (type == ILP_FILTER)
	{
		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				float norm2 = pow(abs(float(r - center.y)), 2) + pow(abs(float(c - center.x)), 2);
				if (sqrt(norm2) < radius)
					hpFilter.at<float>(r, c) = 0;
				else
					hpFilter.at<float>(r, c) = 1;
			}
		}
	}
	//构造巴特沃斯高通滤波器
	if (type == BLP_FILTER)

	{

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				hpFilter.at<float>(r, c) = float(1.0 / (1.0 + pow(radius / sqrt((pow(r - center.y, 2.0) + pow(c - center.x, 2.0))), 2.0 * n)));
			}
		}
	}
	//构造高斯高通滤波
	if (type == GLP_FILTER)
	{

		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				hpFilter.at<float>(r, c) = 1.0 - float(exp(-(pow(c - center.x, 2.0) + pow(r - center.y, 2.0)) / (2 * pow(radius, 2.0))));
			}
		}
	}
	return hpFilter;
}

//回调函数：调整低通滤波的类型，及截断频率
void callback_lpFilter(int, void*)
{
	//构造低通滤波器
	Mat lpFilter = createLPFilter(F.size(), maxLoc, radius, lpType, 2);
	//低通滤波器和图像快速傅里叶变换点乘
	F_lpFilter.create(F.size(), F.type());
	for (int r = 0; r < F_lpFilter.rows; r++)
	{
		for (int c = 0; c < F_lpFilter.cols; c++)
		{
			//分别取出当前位置的快速傅里叶变换和理想低通滤波器的值
			Vec2f F_rc = F.at<Vec2f>(r, c);
			float lpFilter_rc = lpFilter.at<float>(r, c);
			//低通滤波器和图像的快速傅里叶变换对应位置相乘
			F_lpFilter.at<Vec2f>(r, c) = F_rc * lpFilter_rc;
		}
	}

	//低通傅里叶变换的傅里叶谱
	amplitudeSpectrum(F_lpFilter, FlpSpectrum);
	//低通傅里叶谱的灰度级的显示
	FlpSpectrum = graySpectrum(FlpSpectrum);
	namedWindow("低通傅里叶谱", WINDOW_AUTOSIZE);
	imshow("低通傅里叶谱", FlpSpectrum);
	imwrite("FlpSpectrum.jpg", FlpSpectrum);
	//对低通傅里叶变换执行傅里叶逆变换，并只取实部（由于浮点误差造成有微小的虚部，应当忽略）
	Mat result;//低通滤波后的效果
	dft(F_lpFilter, result, DFT_SCALE + DFT_INVERSE + DFT_REAL_OUTPUT);
	///每一个数同乘以(-1)^(r+c)
	for (int r = 0; r < result.rows; r++)
	{
		for (int c = 0; c < result.cols; c++)
		{
			if ((r + c) % 2)
				result.at<float>(r, c) *= -1;
		}
	}
	//将结果转换 CV_8U 类型
	result.convertTo(result, CV_8UC1, 1.0, 0);
	//截取左上部分,大小等于输入图像的大小
	result = result(Rect(0, 0, I.cols, I.rows)).clone();
	namedWindow("经过低通滤波后的图片", WINDOW_AUTOSIZE);
	imshow("经过低通滤波后的图片", result);

}

//回调函数：调整高通滤波的类型，及截断频率
void callback_hpFilter(int, void*)
{
	//构造高通滤波器
	Mat hpFilter = createHPFilter(F.size(), maxLoc, radius, lpType, 2);
	//高通滤波器和图像快速傅里叶变换点乘
	F_hpFilter.create(F.size(), F.type());
	for (int r = 0; r < F_hpFilter.rows; r++)
	{
		for (int c = 0; c < F_hpFilter.cols; c++)
		{
			//分别取出当前位置的快速傅里叶变换和理想低通滤波器的值
			Vec2f F_rc = F.at<Vec2f>(r, c);
			float hpFilter_rc = hpFilter.at<float>(r, c);
			//低通滤波器和图像的快速傅里叶变换对应位置相乘
			F_hpFilter.at<Vec2f>(r, c) = F_rc * hpFilter_rc;
		}
	}

	//高通傅里叶变换的傅里叶谱
	amplitudeSpectrum(F_hpFilter, FhpSpectrum);
	//高通傅里叶谱的灰度级的显示
	FhpSpectrum = graySpectrum(FhpSpectrum);
	namedWindow("高通傅里叶谱", WINDOW_AUTOSIZE);
	imshow("高通傅里叶谱", FhpSpectrum);
	imwrite("FhpSpectrum.jpg", FhpSpectrum);
	//对高通傅里叶变换执行傅里叶逆变换，并只取实部
	Mat result;//高通滤波后的效果
	dft(F_hpFilter, result, DFT_SCALE + DFT_INVERSE + DFT_REAL_OUTPUT);
	//每一个数同乘以(-1)^(x+y)
	for (int r = 0; r < result.rows; r++)
	{
		for (int c = 0; c < result.cols; c++)
		{
			if ((r + c) % 2)
				result.at<float>(r, c) *= -1;
		}
	}
	//将结果转换 CV_8U 类型
	result.convertTo(result, CV_8UC1, 1.0, 0);
	//截取左上部分,大小等于输入图像的大小
	result = result(Rect(0, 0, I.cols, I.rows)).clone();
	namedWindow("经过高通滤波后的图片", WINDOW_AUTOSIZE);
	imshow("经过高通滤波后的图片", result);

}

//低通高通滤波相同处理部分
void Filter(Mat I)
{
	Mat G;
	//色彩空间转换，转换为 灰度模式
	cvtColor(I, G, CV_BGR2GRAY);
	//数据类型转换，转换为 浮点型
	Mat fI;
	G.convertTo(fI, CV_32FC1, 1.0, 0.0);
	//每一个数乘以(-1)^(r+c)，对频谱进行中心化
	for (int r = 0; r < fI.rows; r++)
	{
		for (int c = 0; c < fI.cols; c++)
		{
			if ((r + c) % 2)
				fI.at<float>(r, c) *= -1;
		}
	}
	//补零和快速傅里叶变换
	fft2Image(fI, F);
	//求傅里叶变换的幅度谱
	Mat amplSpec;
	amplitudeSpectrum(F, amplSpec);
	//幅度谱的灰度级显示
	Mat spectrum = graySpectrum(amplSpec);
	namedWindow("原傅里叶谱的灰度级显示", WINDOW_AUTOSIZE);
	imshow("原傅里叶谱的灰度级显示", spectrum);
	//相位谱的灰度级显示

	imwrite("spectrum.jpg", spectrum);
	//找到傅里叶谱中心
	maxLoc.x = amplSpec.cols / 2;  maxLoc.y = amplSpec.rows / 2;
}

//低通滤波
void LowFilter()
{
	/* -- 低通滤波 -- */
	namedWindow("低通傅里叶谱", WINDOW_AUTOSIZE);
	createTrackbar("低通类型:", "低通傅里叶谱", &lpType, MAX_LPTYPE, callback_lpFilter);
	createTrackbar("半径:", "低通傅里叶谱", &radius, Max_RADIUS, callback_lpFilter);
	callback_lpFilter(0, 0);
	waitKey(0);
}

//高通滤波
void HighFilter()
{
	/* -- 高通滤波 -- */
	namedWindow("高通傅里叶谱", WINDOW_AUTOSIZE);
	createTrackbar("高通类型:", "高通傅里叶谱", &lpType, MAX_LPTYPE, callback_hpFilter);
	createTrackbar("半径:", "高通傅里叶谱", &radius, Max_RADIUS, callback_hpFilter);
	callback_hpFilter(0, 0);
	waitKey(0);
}

//回调函数：调整旋转角度和缩放倍数
void callback_Rotate(int, void*)
{
	//double angle = degree * CV_PI / 180.;
	//double alpha = cos(angle);
	//double beta = sin(angle);
	//int iWidth = Img_Rotate.cols;
	//int iHeight = Img_Rotate.rows;
	//int iNewWidth = cvRound(iWidth * fabs(alpha) + iHeight * fabs(beta));
	//int iNewHeight = cvRound(iHeight * fabs(alpha) + iWidth * fabs(beta));

	//double m[6];//旋转矩阵
	//m[0] = alpha;
	//m[1] = beta;
	//m[2] = (1 - alpha) * iWidth / 2. - beta * iHeight / 2.;
	//m[3] = -m[1];
	//m[4] = m[0];
	//m[5] = beta * iWidth / 2. + (1 - alpha) * iHeight / 2.;

	//Mat M = Mat(2, 3, CV_64F, m);
	//Mat matDst1 = Mat(Size(iNewWidth, iNewHeight), Img_Rotate.type(), Scalar::all(0));

	//求M的逆矩阵，即将m变成m的逆
	//double D = m[0] * m[4] - m[1] * m[3];
	//D = D != 0 ? 1. / D : 0;
	//double A11 = m[4] * D, A22 = m[0] * D;
	//m[0] = A11; m[1] *= -D;
	//m[3] *= -D; m[4] = A22;
	//double b1 = -m[0] * m[2] - m[1] * m[5];
	//double b2 = -m[3] * m[2] - m[4] * m[5];
	//m[2] = b1; m[5] = b2;

	//int round_delta = 512;//由于数据扩大了1024倍，此部分相当于对X0和Y0增加0.5
	//for (int y = 0; y < iNewHeight; ++y)
	//{
	//	for (int x = 0; x < iNewWidth; ++x)
	//	{
	//		int adelta = saturate_cast<int>(m[0] * x * 1024);
	//		int bdelta = saturate_cast<int>(m[3] * x * 1024);
	//		int X0 = saturate_cast<int>((m[1] * y + m[2]) * 1024) + round_delta;/*旋转中心移至图像中间*/
	//		int Y0 = saturate_cast<int>((m[4] * y + m[5]) * 1024) + round_delta;
	//		int X = (X0 + adelta) >> 10;
	//		int Y = (Y0 + bdelta) >> 10;

	//		if ((unsigned)X < iWidth && (unsigned)Y < iHeight)
	//		{
	//			matDst1.at<Vec3b>(y, x) = Img_Rotate.at<Vec3b>(Y, X);//Img_Rotate对应逆旋转操作后的像素点
	//		}
	//	}
	//}

	Mat rot_mat(2, 3, CV_64F);
	Mat matDst1;

	//计算关于图像中心的旋转矩阵
	Point center = Point(Img_Rotate.cols / 2, Img_Rotate.rows / 2);

	//根据以上参数得到旋转矩阵
	rot_mat = getRotationMatrix2D(center, degree, (double)scale);

	//计算旋转后的画布大小，并将旋转中心平移到新的旋转中心
	Rect bbox = RotatedRect(center, Size(Img_Rotate.cols * (double)scale, Img_Rotate.rows * (double)scale), degree).boundingRect();
	rot_mat.at<double>(0, 2) += bbox.width / 2.0 - center.x;
	rot_mat.at<double>(1, 2) += bbox.height / 2.0 - center.y;

	//旋转图像
	warpAffine(Img_Rotate, matDst1, rot_mat, bbox.size());

	//显示旋转效果
	imshow("图像旋转和缩放", matDst1);
	waitKey(0);

}

//图像旋转和缩放
void Rotate()
{
	namedWindow("图像旋转和缩放", WINDOW_AUTOSIZE);
	createTrackbar("旋转角度 :", "图像旋转和缩放", &degree, MAX_DEGREE, callback_Rotate);
	createTrackbar("缩放倍数 :", "图像旋转和缩放", &scale, MAX_SCALE, callback_Rotate);
	waitKey(0);
}

//回调函数：调整平移距离
void callback_Translation(int, void*)
{
	Size dst_sz = Img_Translation.size();
	Mat matDst2;

	//定义平移矩阵
	Mat t_mat = Mat::zeros(2, 3, CV_32FC1);

	t_mat.at<float>(0, 0) = 1;
	t_mat.at<float>(0, 2) = dy; //水平平移量
	t_mat.at<float>(1, 1) = 1;
	t_mat.at<float>(1, 2) = dx; //竖直平移量

	//根据平移矩阵进行仿射变换
	warpAffine(Img_Translation, matDst2, t_mat, Size(Img_Translation.cols + dy, Img_Translation.rows + dx));

	//显示平移效果
	imshow("图像平移", matDst2);
	waitKey(0);
}

//图像平移
void Translation() 
{
	//图像平移
	namedWindow("图像平移", WINDOW_AUTOSIZE);
	createTrackbar("X方向平移量 :", "图像平移", &dx, MAX_DX, callback_Translation);
	createTrackbar("Y方向平移量 :", "图像平移", &dy, MAX_DY, callback_Translation);
	callback_Translation(0, 0);
	waitKey(0);
}

//色彩平衡
void equalizeCallback(Mat img)
{
	//通过使用直方图均衡化，可以增加对比度，并提升暴露过度或暴露不足的细节
	Mat result;
	Mat ycrcb;
	//用cvtColor函数将BGR图像转化为YCrCb颜色格式
	cvtColor(img, ycrcb, COLOR_BGR2YCrCb);
	//YUV三通道分离
	vector<Mat> channels;
	split(ycrcb, channels);
	// 用Y通道（亮度）进行直方图均衡
	equalizeHist(channels[0], channels[0]);
	//合成生成的通道并将其转换为BGR格式
	merge(channels, ycrcb);
	cvtColor(ycrcb, result, COLOR_YCrCb2BGR);
	// 新旧对比
	imshow("原图", img);
	imshow("色彩平衡结果", result);
	waitKey(0);
}

int main()
{
	Mat src, dst;
	src = imread("20180620-tianjin.bmp");
	I = src.clone();

	if (!src.data)
	{
		cout << "could not load image" << endl;
		return -1;
	}

	//必做题1：灰度线性变换
	Mat gray = src.clone();
	LinearProc(gray);

	//必做题2：中值滤波
	salt(src, 3000);//加入3000个盐噪声255  
	pepper(src, 3000);//加入3000个椒噪声0  
	// 加入椒盐噪声的图像  
	imshow(" 加入椒盐噪声的图像 ", src);
	//	窗口大小3*3的中值滤波
	medianBlur(src, dst, 3);
	//	中值滤波后的图像
	imshow("中值滤波结果", dst);
	waitKey(0);

	//必做题2：低通&高通滤波
	Filter(I);
	HighFilter();
	LowFilter();

	//选做题1：旋转和缩放
	Img_Rotate=imread("123.jpg");
	Rotate();
	//选做题1：平移
	Img_Translation= imread("123.jpg");
	Translation();

	//选做题5：色彩平衡
	Mat clr = imread("123.jpg");
	equalizeCallback(clr);

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
