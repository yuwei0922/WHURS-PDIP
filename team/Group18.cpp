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

using namespace std;  //ʡȥ��Ļ�������coutǰ��std::
using namespace cv;   // ʡȥ����ǰ���cv::�ı�Ҫ��

void GaussFilter(Mat grayimg);//��˹ƽ��
int factorial(int n);//����׳�
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

//��˹ƽ��
void GaussFilter(Mat grayimg)
{
	unsigned char* pimg = grayimg.data;//ͼ���ַ
	int height = grayimg.rows;
	int width = grayimg.cols;
	//Ϊ��ͼ�����ַ���洢�ռ䣨�������߽��
	Mat imgnew;
	imgnew.create(height, width, CV_8UC1);
	unsigned char* pnewimg = imgnew.data;
	for (int i = 1; i < height - 1; i++)
	{
		for (int j = 1; j < width - 1; j++)
		{
			pnewimg[i * width + j] = 0;
		}
	}//��ʼ��

	 //��ȡ������txt�е��˲�����
	FILE* fp;
	int n = 0;
	fp = fopen("gausslvbo.txt", "r");

	if (!fp)
	{
		printf("�޷���ȡ�˲�����\n");
	}

	double H[9];//�洢����
	for (int i = 0; i < 9; i++)
	{
		fscanf_s(fp, "%lf", &H[i]);
	}

	//�������
	for (int i = 1; i < height - 1; i++)
	{ //��1��ʼ���Ȳ�����߽��
		for (int j = 1; j < width - 1; j++)
		{
			double t = 0;
			for (int m = 0; m < 3; m++)
			{//ѡȡ��ѡ��8-����ĵ�
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
	//�ٶԱ߽����д���(����
	int t = 0;//��һ��
	for (int q = 0; q < width; q++)
	{
		pnewimg[t * width + q] = pimg[t * width + q];
	}
	t = height - 1;//���һ��
	for (int q = 0; q < width; q++)
	{
		pnewimg[t * width + q] = pimg[t * width + q];
	}
	int q = 0;//��һ��
	for (int t = 0; t < height; t++)
	{
		pnewimg[t * width + q] = pimg[t * width + q];
	}
	q = width - 1;//���һ��
	for (int t = 0; t < height; t++)
	{
		pnewimg[t * width + q] = pimg[t * width + q];
	}

	namedWindow("��˹ƽ��", WINDOW_NORMAL);
	imshow("��˹ƽ��", imgnew); //��ʾ�任���ͼ��
	imwrite("smoothimg.bmp", imgnew);//����ͼƬ
	waitKey();
}


/**********************Sobel����*************************/
//�׳�
int factorial(int n) 
{
	int fac = 1;
	//0�Ľ׳�
	if (n == 0)
		return fac;
	for (int i = 1; i <= n; ++i) 
    {
		fac *= i;
	}
	return fac;
}

//���Sobelƽ������
Mat getSobelSmoooth(int wsize) 
{
	int n = wsize - 1;
	Mat SobelSmooothoper = Mat::zeros(Size(wsize, 1), CV_32FC1);
	for (int k = 0; k <= n; k++) 
    {
		float* pt = SobelSmooothoper.ptr<float>(0);//ָ���һ�е�һ��Ԫ��
		pt[k] = factorial(n) / (factorial(k) * factorial(n - k));
	}
	return SobelSmooothoper;
}

//���Sobel�������
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

//�ɷ������������ȴ�ֱ����������ˮƽ������
void sepConv2D_Y_X(Mat& src, Mat& dst, Mat kernel_Y, Mat kernel_X, int ddepth, Point anchor , int delta, int borderType ) 
{
	Mat dst_kernel_Y;
    filter2D(src, dst_kernel_Y, ddepth, kernel_Y, anchor, delta, borderType); //��ֱ������
    filter2D(dst_kernel_Y, dst, ddepth, kernel_X, anchor, delta, borderType); //ˮƽ������
}

//�ɷ�������������ˮƽ����������ֱ������
void sepConv2D_X_Y(Mat& src, Mat& dst, Mat kernel_X, Mat kernel_Y, int ddepth, Point anchor , int delta , int borderType ) 
{
	Mat dst_kernel_X;
    filter2D(src, dst_kernel_X, ddepth, kernel_X, anchor, delta, borderType); //ˮƽ������
    filter2D(dst_kernel_X, dst, ddepth, kernel_Y, anchor, delta, borderType); //��ֱ������
}

/*************************************************************************************************/

//Sobel���ӱ�Ե���
//dst_X ��ֱ����
//dst_Y ˮƽ����
void Sobel(Mat& src, Mat& dst_X, Mat& dst_Y, Mat& dst, int wsize, int ddepth, Point anchor , int delta, int borderType )
{
	Mat SobelSmooothoper = getSobelSmoooth(wsize); //ƽ��ϵ��
	Mat Sobeldiffoper = getSobeldiff(wsize); //���ϵ��

    //�ɷ������������ȴ�ֱ����ƽ������ˮƽ�����֡����õ���ֱ��Ե
	sepConv2D_Y_X(src, dst_X, SobelSmooothoper.t(), Sobeldiffoper, ddepth);

	//�ɷ�������������ˮƽ����ƽ������ֱ�����֡����õ�ˮƽ��Ե
	sepConv2D_X_Y(src, dst_Y, SobelSmooothoper, Sobeldiffoper.t(), ddepth);

	//��Եǿ�ȣ����ƣ�
	dst = abs(dst_X) + abs(dst_Y);
	convertScaleAbs(dst, dst); //�����ֵ��תΪ�޷���8λͼ
}

//ȷ��һ����������Ƿ���ͼ����
bool checkInRang(int r, int c, int rows, int cols) 
{
	if (r >= 0 && r < rows && c >= 0 && c < cols)
		return true;
	else
		return false;
}

//��ȷ����Ե��������ӳ���Ե
void trace(Mat& edgeMag_noMaxsup, Mat& edge, float TL, int r, int c, int rows, int cols) 
{
	if (edge.at<uchar>(r, c) == 0) 
    {
		edge.at<uchar>(r, c) = 255;
		for (int i = -1; i <= 1; ++i) //�����жϸñ�Ե��3*3�������Ƿ��д��ڸ���ֵ�ĵ�
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

//Canny��Ե���
void Edge_Canny(Mat& src, Mat& edge, float TL, float TH, int wsize , bool L2graydient ) 
{

	int rows = src.rows;
	int cols = src.cols;

	//sobel����
	Mat dx, dy, sobel_dst;
	Sobel(src, dx, dy, sobel_dst, wsize, CV_32FC1);

	//�����ݶȷ�ֵ
	Mat edgeMag;
	if (L2graydient)
		magnitude(dx, dy, edgeMag); //��ƽ��
	else
		edgeMag = abs(dx) + abs(dy); //����ֵ֮�ͽ���

	edgeMag.convertTo(edgeMag, CV_8UC1, 1.0, 0);//�����ת�� CV_8U ����
	namedWindow("�ݶȷ�ֵ", WINDOW_NORMAL);
	imshow("�ݶȷ�ֵ", edgeMag); //��ʾ�任���ͼ��
	waitKey();

	//�����ݶȷ��� �Լ� �Ǽ���ֵ����
	edgeMag.convertTo(edgeMag, CV_32FC1, 1.0, 0);
	Mat edgeMag_noMaxsup = Mat::zeros(rows, cols, CV_32FC1);
	for (int r = 1; r < rows - 1; ++r) 
    {
		for (int c = 1; c < cols - 1; ++c) 
        {
			float x = dx.at<float>(r, c);
			float y = dy.at<float>(r, c);
			float angle = std::atan2f(y, x) / CV_PI * 180; //��ǰλ���ݶȷ���
			float mag = edgeMag.at<float>(r, c);  //��ǰλ���ݶȷ�ֵ

	        //�Ǽ���ֵ����
			//�ݶȷ���Ϊˮƽ����-3*3���������ҷ���Ƚ�
			if (abs(angle) < 22.5 || abs(angle) > 157.5)
            {
				float left = edgeMag.at<float>(r, c - 1);
				float right = edgeMag.at<float>(r, c + 1);
				if (mag >= left && mag >= right)
					edgeMag_noMaxsup.at<float>(r, c) = mag;
			}

			//�ݶȷ���Ϊ��ֱ����-3*3���������·���Ƚ�
			if ((angle >= 67.5 && angle <= 112.5) || (angle >= -112.5 && angle <= -67.5))
            {
				float top = edgeMag.at<float>(r - 1, c);
				float down = edgeMag.at<float>(r + 1, c);
				if (mag >= top && mag >= down)
					edgeMag_noMaxsup.at<float>(r, c) = mag;
			}

			//�ݶȷ���Ϊ-45��-3*3�������������·���Ƚ�
			if ((angle > 112.5 && angle <= 157.5) || (angle > -67.5 && angle <= -22.5))
            {
				float right_top = edgeMag.at<float>(r - 1, c + 1);
				float left_down = edgeMag.at<float>(r + 1, c - 1);
				if (mag >= right_top && mag >= left_down)
					edgeMag_noMaxsup.at<float>(r, c) = mag;
			}

			//�ݶȷ���Ϊ+45��-3*3�������������Ϸ���Ƚ�
			if ((angle >= 22.5 && angle < 67.5) || (angle >= -157.5 && angle < -112.5))
            {
				float left_top = edgeMag.at<float>(r - 1, c - 1);
				float right_down = edgeMag.at<float>(r + 1, c + 1);
				if (mag >= left_top && mag >= right_down)
					edgeMag_noMaxsup.at<float>(r, c) = mag;
			}
		}
	}
	edgeMag_noMaxsup.convertTo(edgeMag_noMaxsup, CV_8UC1, 1.0, 0);//�����ת�� CV_8U ����
	namedWindow("�Ǽ���ֵ����", WINDOW_NORMAL);
	imshow("�Ǽ���ֵ����", edgeMag_noMaxsup); //��ʾ�任���ͼ��
	waitKey();

	//˫��ֵ������Ե����
	edgeMag_noMaxsup.convertTo(edgeMag_noMaxsup, CV_32FC1, 1.0, 0);
	edge = Mat::zeros(rows, cols, CV_8UC1);
	for (int r = 1; r < rows - 1; ++r) 
    {
		for (int c = 1; c < cols - 1; ++c)  
        {
			float mag = edgeMag_noMaxsup.at<float>(r, c);
			//���ڸ���ֵ��ȷ��Ϊ��Ե��
			if (mag >= TH)
				trace(edgeMag_noMaxsup, edge, TL, r, c, rows, cols);
            //С�ڵ���ֵ���ų�Ϊ��Ե��
			else if (mag < TL)
				edge.at<uchar>(r, c) = 0;
		}
	}

}


/***************************Hough�任****************************/
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

	image = img.ptr();    //�õ�ͼ���ָ��
	step = img.step;    //�õ�ͼ��Ĳ���
	width = img.cols;    //�õ�ͼ��Ŀ�
	height = img.rows;    //�õ�ͼ��ĸ�
	//�ɽǶȺ;���ķֱ��ʵõ��ǶȺ;����������������任��ǶȺ;���ĸ���
	numangle = cvRound(CV_PI / theta);
	numrho = cvRound(((width + height) * 2 + 1) / rho);

	_accum.allocate((numangle + 2) * (numrho + 2));
	//Ϊ������������ڴ�ռ�
	_sort_buf.allocate(numangle * numrho);
	//Ϊ���Һ������б�����ڴ�ռ�
	_tabSin.allocate(numangle);
	_tabCos.allocate(numangle);
	//�ֱ��������ڴ�ռ�ĵ�ַָ��
	int* accum = _accum, * sort_buf = _sort_buf;
	float* tabSin = _tabSin, * tabCos = _tabCos;
	//�ۼ�����������
	memset(accum, 0, sizeof(accum[0]) * (numangle + 2) * (numrho + 2));

	float ang = 0;
	//Ϊ�����ظ����㣬���ȼ����sin��i/�Ѻ�cos��i/��
	for (int n = 0; n < numangle; ang += theta, n++)
	{
		tabSin[n] = (float)(sin((double)ang) * irho);
		tabCos[n] = (float)(cos((double)ang) * irho);
	}

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
		{
			//ֻ��ͼ��ķ���ֵ������ֻ��ͼ��ı�Ե���ؽ��л���任
			if (image[i * step + j] != 0)
				for (int n = 0; n < numangle; n++)
				{
					int r = cvRound(j * tabCos[n] + i * tabSin[n]);
					r += (numrho - 1) / 2;
					//r��ʾ���Ǿ��룬n��ʾ���ǽǵ㣬���ۼ������ҵ���������Ӧ��λ�ã�������ռ��ڵ�λ�ã�����ֵ��1
					accum[(n + 1) * (numrho + 2) + r + 1]++;
				}
		}

	for (int r = 0; r < numrho; r++)
		for (int n = 0; n < numangle; n++)
		{
			//�õ���ǰֵ���ۼ��������λ��
			int base = (n + 1) * (numrho + 2) + r + 1;
			if (accum[base] > threshold &&    //������������õ���ֵ
				//��4�����ڽ��зǼ���ֵ����
				accum[base] > accum[base - 1] && accum[base] >= accum[base + 1] &&
				accum[base] > accum[base - numrho - 2] && accum[base] >= accum[base + numrho + 2])
				//�Ѽ���ֵλ�ô������������ڡ���sort_buf
				sort_buf[total++] = base;
		}

	//���ȶ���һ���߶�
	scale = 1. / (numrho + 2);
	vector<Vec2f> lines;
	for (i = 0; i < total; i++)
	{
		Vec2f temp;
		//idxΪ����ֵ���ۼ��������λ��
		int idx = sort_buf[i];
		//������ü���ֵ�ڻ���ռ��е�λ��
		int n = cvFloor(idx * scale) - 1;
		int r = idx - (n + 1) * (numrho + 2) - 1;
		//���յõ�����ֵ����Ӧ�ĽǶȺ;���
		temp[0] = (r - (numrho - 1) * 0.5f) * rho;
		temp[1] = n * theta;
		//�洢��������
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
    Mat image_input = imread("20180620-tianjin.bmp", IMREAD_ANYCOLOR);   // ����ͼƬ 

    if (image_input.empty())     // �ж��ļ��Ƿ�������  
    {
        fprintf(stderr, "Can not load image %s\n", "20180620-tianjin.bmp");
        waitKey(6000);  // �ȴ�6000 ms�󴰿��Զ��ر�   
        return -1;
    }
	namedWindow("ԭͼ", WINDOW_NORMAL);
	imshow("ԭͼ", image_input); //��ʾԭͼ

	Mat image_gray;
	cvtColor(image_input, image_gray, COLOR_RGB2GRAY);
	GaussFilter(image_gray);//��˹ƽ��

    Mat gray = imread("smoothimg.bmp", IMREAD_GRAYSCALE);
    Edge_Canny(gray, image_gray, 80, 160, 3, 1 );//����һ��canny��Ե���

    namedWindow("Edge", WINDOW_NORMAL);
    imshow("Edge", image_gray); //��ʾCanny��Ե�����ͼ��
  	Mat image_output(image_gray.rows, image_gray.cols, CV_8U, Scalar(255));
	image_input.copyTo(image_output);
	LineFinder finder;
	finder.setthreshold(130);//Hough�任
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
