// DIP.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>   
#include <opencv2\imgproc\imgproc.hpp>  
#include <cstdlib>

using namespace std;  //ʡȥ��Ļ�������coutǰ��std::
using namespace cv;   // ʡȥ����ǰ���cv::�ı�Ҫ��

//���������
void salt(Mat image, int n)
{
	int i, j;
	for (int k = 0; k < n; k++)
	{
		// rand()�������������
		i = rand() % image.cols;
		j = rand() % image.rows;
		if (image.type() == CV_8UC1)
		{ // �Ҷ�ͼ��
			image.at<uchar>(j, i) = 255;
		}
		else if (image.type() == CV_8UC3)
		{ // ��ɫͼ��
			image.at<cv::Vec3b>(j, i)[0] = 255;
			image.at<cv::Vec3b>(j, i)[1] = 255;
			image.at<cv::Vec3b>(j, i)[2] = 255;
		}
	}
}

//��ӽ�����
void pepper(Mat image, int n)
{
	int i, j;
	for (int k = 0; k < n; k++)
	{
		// rand()�������������
		i = rand() % image.cols;
		j = rand() % image.rows;
		if (image.type() == CV_8UC1)
		{ // �Ҷ�ͼ��
			image.at<uchar>(j, i) = 0;
		}
		else if (image.type() == CV_8UC3)
		{ // ��ɫͼ��
			image.at<cv::Vec3b>(j, i)[0] = 0;
			image.at<cv::Vec3b>(j, i)[1] = 0;
			image.at<cv::Vec3b>(j, i)[2] = 0;
		}
	}
}

int radius = 20;//�ض�Ƶ��
int lpType = 0;//��ͨ�˲���������
const int Max_RADIUS = 100;//�������Ľض�Ƶ��
const int MAX_LPTYPE = 2;//�����˲�����

Mat I;//��ͨ&��ͨ�˲��ֲ�����
Mat F;//ͼ��Ŀ��ٸ���Ҷ�任
Mat FlpSpectrum;//��ͨ����Ҷ�任�ĸ���Ҷ�׻Ҷȼ�
Mat FhpSpectrum;//��ͨ����Ҷ�任�ĸ���Ҷ�׻Ҷȼ�
Mat F_lpFilter;//��ͨ����Ҷ�任
Mat F_hpFilter;//��ͨ����Ҷ�任

Point maxLoc;	//Point maxLocΪ����Ҷ�׵����ֵ������

Mat Img_Rotate; //ͼ����ת
int degree = 0;//��ת�Ƕ�
int scale = 1;//���ű���
const int MAX_DEGREE = 360; //����������ת�Ƕ�
const int MAX_SCALE = 5; //�����������ű���

Mat Img_Translation; //ͼ����ת
int dx = 0;//x����
int dy = 0;//y����
const int MAX_DX = 500; //����x��������ƽ�ƾ���
const int MAX_DY = 500; //����y��������ƽ�ƾ���

//�Ҷ����Ա任
void LinearProc(Mat src)
{
	namedWindow("ԭͼ", CV_WINDOW_AUTOSIZE);
	imshow("ԭͼ", src);
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
				//��ɫͼ��
				int b = src.at<Vec3b>(row, col)[0];//��ȡ����ͨ���Ҷ�ֵ
				int g = src.at<Vec3b>(row, col)[1];
				int r = src.at<Vec3b>(row, col)[2];
				dst.at<Vec3b>(row, col)[0] = saturate_cast<uchar>((alpha * b + beta));//���Ա任
				dst.at<Vec3b>(row, col)[1] = saturate_cast<uchar>((alpha * g + beta));
				dst.at<Vec3b>(row, col)[2] = saturate_cast<uchar>((alpha * r + beta));
			}
			else if (src.channels() == 1) 
			{
				//�Ҷ�ͼ��
				int v = src.at<uchar>(row, col);
				dst.at<uchar>(row, col) = saturate_cast<uchar>(alpha * v + beta);
			}
		}
	}

	imshow("�Ҷ����Ա任���", dst);
	waitKey(0);
}

//���ٸ���Ҷ�任
void fft2Image(InputArray _src, OutputArray _dst)
{
	//��InputArray��������ת����Mat����
	Mat src = _src.getMat();
	//�ж�λ���ͨ��
	CV_Assert(src.type() == CV_32FC1 || src.type() == CV_64FC1);
	CV_Assert(src.channels() == 1 || src.channels() == 2);
	int rows = src.rows;
	int cols = src.cols;
	//Ϊ�˽��п��ٵĸ���Ҷ�任�����к��е�����,�ҵ����������ֵ
	Mat padded;
	int rPadded = getOptimalDFTSize(rows);
	int cPadded = getOptimalDFTSize(cols);
	//���б�Ե����,����ֵΪ��
	copyMakeBorder(src, padded, 0, rPadded - rows, 0, cPadded - cols, BORDER_CONSTANT, Scalar::all(0));
	//���ٵĸ���Ҷ�任��˫ͨ�������ڴ洢ʵ�� �� �鲿��
	dft(padded, _dst, DFT_COMPLEX_OUTPUT);
}

//������
void amplitudeSpectrum(InputArray _srcFFT, OutputArray _dstSpectrum)
{
	//�жϸ���Ҷ�任������ͨ����ʵ�����鲿��
	CV_Assert(_srcFFT.channels() == 2);
	//����ͨ��
	vector<Mat> FFT2Channel;
	split(_srcFFT, FFT2Channel);
	//���㸵��Ҷ�任�ķ����� sqrt(pow(R,2)+pow(I,2))
	magnitude(FFT2Channel[0], FFT2Channel[1], _dstSpectrum);
}

//�����׵ĻҶȼ���ʾ
Mat graySpectrum(Mat spectrum)
{
	Mat dst;
	log(spectrum + 1, dst);
	//��һ��
	normalize(dst, dst, 0, 1, NORM_MINMAX);
	//Ϊ�˽��лҶȼ���ʾ��������ת��
	dst.convertTo(dst, CV_8UC1, 255, 0);
	return dst;
}

//�����ͨ�˲���
enum LPFILTER_TYPE { ILP_FILTER = 0, BLP_FILTER = 1, GLP_FILTER = 2 };
Mat createLPFilter(Size size, Point center, float radius, int type, int n = 2)
{
	Mat lpFilter = Mat::zeros(size, CV_32FC1);
	int rows = size.height;
	int cols = size.width;
	if (radius <= 0)
		return lpFilter;
	//���������ͨ�˲���
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
	//������װ�����˹��ͨ�˲���
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
	//�����˹��ͨ�˲�
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

//�����ͨ�˲���
Mat createHPFilter(Size size, Point center, float radius, int type, int n = 2)
{
	Mat hpFilter = Mat::zeros(size, CV_32FC1);
	int rows = size.height;
	int cols = size.width;
	if (radius <= 0)
		return hpFilter;

	//���������ͨ�˲���
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
	//���������˹��ͨ�˲���
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
	//�����˹��ͨ�˲�
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

//�ص�������������ͨ�˲������ͣ����ض�Ƶ��
void callback_lpFilter(int, void*)
{
	//�����ͨ�˲���
	Mat lpFilter = createLPFilter(F.size(), maxLoc, radius, lpType, 2);
	//��ͨ�˲�����ͼ����ٸ���Ҷ�任���
	F_lpFilter.create(F.size(), F.type());
	for (int r = 0; r < F_lpFilter.rows; r++)
	{
		for (int c = 0; c < F_lpFilter.cols; c++)
		{
			//�ֱ�ȡ����ǰλ�õĿ��ٸ���Ҷ�任�������ͨ�˲�����ֵ
			Vec2f F_rc = F.at<Vec2f>(r, c);
			float lpFilter_rc = lpFilter.at<float>(r, c);
			//��ͨ�˲�����ͼ��Ŀ��ٸ���Ҷ�任��Ӧλ�����
			F_lpFilter.at<Vec2f>(r, c) = F_rc * lpFilter_rc;
		}
	}

	//��ͨ����Ҷ�任�ĸ���Ҷ��
	amplitudeSpectrum(F_lpFilter, FlpSpectrum);
	//��ͨ����Ҷ�׵ĻҶȼ�����ʾ
	FlpSpectrum = graySpectrum(FlpSpectrum);
	namedWindow("��ͨ����Ҷ��", WINDOW_AUTOSIZE);
	imshow("��ͨ����Ҷ��", FlpSpectrum);
	imwrite("FlpSpectrum.jpg", FlpSpectrum);
	//�Ե�ͨ����Ҷ�任ִ�и���Ҷ��任����ֻȡʵ�������ڸ�����������΢С���鲿��Ӧ�����ԣ�
	Mat result;//��ͨ�˲����Ч��
	dft(F_lpFilter, result, DFT_SCALE + DFT_INVERSE + DFT_REAL_OUTPUT);
	///ÿһ����ͬ����(-1)^(r+c)
	for (int r = 0; r < result.rows; r++)
	{
		for (int c = 0; c < result.cols; c++)
		{
			if ((r + c) % 2)
				result.at<float>(r, c) *= -1;
		}
	}
	//�����ת�� CV_8U ����
	result.convertTo(result, CV_8UC1, 1.0, 0);
	//��ȡ���ϲ���,��С��������ͼ��Ĵ�С
	result = result(Rect(0, 0, I.cols, I.rows)).clone();
	namedWindow("������ͨ�˲����ͼƬ", WINDOW_AUTOSIZE);
	imshow("������ͨ�˲����ͼƬ", result);

}

//�ص�������������ͨ�˲������ͣ����ض�Ƶ��
void callback_hpFilter(int, void*)
{
	//�����ͨ�˲���
	Mat hpFilter = createHPFilter(F.size(), maxLoc, radius, lpType, 2);
	//��ͨ�˲�����ͼ����ٸ���Ҷ�任���
	F_hpFilter.create(F.size(), F.type());
	for (int r = 0; r < F_hpFilter.rows; r++)
	{
		for (int c = 0; c < F_hpFilter.cols; c++)
		{
			//�ֱ�ȡ����ǰλ�õĿ��ٸ���Ҷ�任�������ͨ�˲�����ֵ
			Vec2f F_rc = F.at<Vec2f>(r, c);
			float hpFilter_rc = hpFilter.at<float>(r, c);
			//��ͨ�˲�����ͼ��Ŀ��ٸ���Ҷ�任��Ӧλ�����
			F_hpFilter.at<Vec2f>(r, c) = F_rc * hpFilter_rc;
		}
	}

	//��ͨ����Ҷ�任�ĸ���Ҷ��
	amplitudeSpectrum(F_hpFilter, FhpSpectrum);
	//��ͨ����Ҷ�׵ĻҶȼ�����ʾ
	FhpSpectrum = graySpectrum(FhpSpectrum);
	namedWindow("��ͨ����Ҷ��", WINDOW_AUTOSIZE);
	imshow("��ͨ����Ҷ��", FhpSpectrum);
	imwrite("FhpSpectrum.jpg", FhpSpectrum);
	//�Ը�ͨ����Ҷ�任ִ�и���Ҷ��任����ֻȡʵ��
	Mat result;//��ͨ�˲����Ч��
	dft(F_hpFilter, result, DFT_SCALE + DFT_INVERSE + DFT_REAL_OUTPUT);
	//ÿһ����ͬ����(-1)^(x+y)
	for (int r = 0; r < result.rows; r++)
	{
		for (int c = 0; c < result.cols; c++)
		{
			if ((r + c) % 2)
				result.at<float>(r, c) *= -1;
		}
	}
	//�����ת�� CV_8U ����
	result.convertTo(result, CV_8UC1, 1.0, 0);
	//��ȡ���ϲ���,��С��������ͼ��Ĵ�С
	result = result(Rect(0, 0, I.cols, I.rows)).clone();
	namedWindow("������ͨ�˲����ͼƬ", WINDOW_AUTOSIZE);
	imshow("������ͨ�˲����ͼƬ", result);

}

//��ͨ��ͨ�˲���ͬ������
void Filter(Mat I)
{
	Mat G;
	//ɫ�ʿռ�ת����ת��Ϊ �Ҷ�ģʽ
	cvtColor(I, G, CV_BGR2GRAY);
	//��������ת����ת��Ϊ ������
	Mat fI;
	G.convertTo(fI, CV_32FC1, 1.0, 0.0);
	//ÿһ��������(-1)^(r+c)����Ƶ�׽������Ļ�
	for (int r = 0; r < fI.rows; r++)
	{
		for (int c = 0; c < fI.cols; c++)
		{
			if ((r + c) % 2)
				fI.at<float>(r, c) *= -1;
		}
	}
	//����Ϳ��ٸ���Ҷ�任
	fft2Image(fI, F);
	//����Ҷ�任�ķ�����
	Mat amplSpec;
	amplitudeSpectrum(F, amplSpec);
	//�����׵ĻҶȼ���ʾ
	Mat spectrum = graySpectrum(amplSpec);
	namedWindow("ԭ����Ҷ�׵ĻҶȼ���ʾ", WINDOW_AUTOSIZE);
	imshow("ԭ����Ҷ�׵ĻҶȼ���ʾ", spectrum);
	//��λ�׵ĻҶȼ���ʾ

	imwrite("spectrum.jpg", spectrum);
	//�ҵ�����Ҷ������
	maxLoc.x = amplSpec.cols / 2;  maxLoc.y = amplSpec.rows / 2;
}

//��ͨ�˲�
void LowFilter()
{
	/* -- ��ͨ�˲� -- */
	namedWindow("��ͨ����Ҷ��", WINDOW_AUTOSIZE);
	createTrackbar("��ͨ����:", "��ͨ����Ҷ��", &lpType, MAX_LPTYPE, callback_lpFilter);
	createTrackbar("�뾶:", "��ͨ����Ҷ��", &radius, Max_RADIUS, callback_lpFilter);
	callback_lpFilter(0, 0);
	waitKey(0);
}

//��ͨ�˲�
void HighFilter()
{
	/* -- ��ͨ�˲� -- */
	namedWindow("��ͨ����Ҷ��", WINDOW_AUTOSIZE);
	createTrackbar("��ͨ����:", "��ͨ����Ҷ��", &lpType, MAX_LPTYPE, callback_hpFilter);
	createTrackbar("�뾶:", "��ͨ����Ҷ��", &radius, Max_RADIUS, callback_hpFilter);
	callback_hpFilter(0, 0);
	waitKey(0);
}

//�ص�������������ת�ǶȺ����ű���
void callback_Rotate(int, void*)
{
	//double angle = degree * CV_PI / 180.;
	//double alpha = cos(angle);
	//double beta = sin(angle);
	//int iWidth = Img_Rotate.cols;
	//int iHeight = Img_Rotate.rows;
	//int iNewWidth = cvRound(iWidth * fabs(alpha) + iHeight * fabs(beta));
	//int iNewHeight = cvRound(iHeight * fabs(alpha) + iWidth * fabs(beta));

	//double m[6];//��ת����
	//m[0] = alpha;
	//m[1] = beta;
	//m[2] = (1 - alpha) * iWidth / 2. - beta * iHeight / 2.;
	//m[3] = -m[1];
	//m[4] = m[0];
	//m[5] = beta * iWidth / 2. + (1 - alpha) * iHeight / 2.;

	//Mat M = Mat(2, 3, CV_64F, m);
	//Mat matDst1 = Mat(Size(iNewWidth, iNewHeight), Img_Rotate.type(), Scalar::all(0));

	//��M������󣬼���m���m����
	//double D = m[0] * m[4] - m[1] * m[3];
	//D = D != 0 ? 1. / D : 0;
	//double A11 = m[4] * D, A22 = m[0] * D;
	//m[0] = A11; m[1] *= -D;
	//m[3] *= -D; m[4] = A22;
	//double b1 = -m[0] * m[2] - m[1] * m[5];
	//double b2 = -m[3] * m[2] - m[4] * m[5];
	//m[2] = b1; m[5] = b2;

	//int round_delta = 512;//��������������1024�����˲����൱�ڶ�X0��Y0����0.5
	//for (int y = 0; y < iNewHeight; ++y)
	//{
	//	for (int x = 0; x < iNewWidth; ++x)
	//	{
	//		int adelta = saturate_cast<int>(m[0] * x * 1024);
	//		int bdelta = saturate_cast<int>(m[3] * x * 1024);
	//		int X0 = saturate_cast<int>((m[1] * y + m[2]) * 1024) + round_delta;/*��ת��������ͼ���м�*/
	//		int Y0 = saturate_cast<int>((m[4] * y + m[5]) * 1024) + round_delta;
	//		int X = (X0 + adelta) >> 10;
	//		int Y = (Y0 + bdelta) >> 10;

	//		if ((unsigned)X < iWidth && (unsigned)Y < iHeight)
	//		{
	//			matDst1.at<Vec3b>(y, x) = Img_Rotate.at<Vec3b>(Y, X);//Img_Rotate��Ӧ����ת����������ص�
	//		}
	//	}
	//}

	Mat rot_mat(2, 3, CV_64F);
	Mat matDst1;

	//�������ͼ�����ĵ���ת����
	Point center = Point(Img_Rotate.cols / 2, Img_Rotate.rows / 2);

	//�������ϲ����õ���ת����
	rot_mat = getRotationMatrix2D(center, degree, (double)scale);

	//������ת��Ļ�����С��������ת����ƽ�Ƶ��µ���ת����
	Rect bbox = RotatedRect(center, Size(Img_Rotate.cols * (double)scale, Img_Rotate.rows * (double)scale), degree).boundingRect();
	rot_mat.at<double>(0, 2) += bbox.width / 2.0 - center.x;
	rot_mat.at<double>(1, 2) += bbox.height / 2.0 - center.y;

	//��תͼ��
	warpAffine(Img_Rotate, matDst1, rot_mat, bbox.size());

	//��ʾ��תЧ��
	imshow("ͼ����ת������", matDst1);
	waitKey(0);

}

//ͼ����ת������
void Rotate()
{
	namedWindow("ͼ����ת������", WINDOW_AUTOSIZE);
	createTrackbar("��ת�Ƕ� :", "ͼ����ת������", &degree, MAX_DEGREE, callback_Rotate);
	createTrackbar("���ű��� :", "ͼ����ת������", &scale, MAX_SCALE, callback_Rotate);
	waitKey(0);
}

//�ص�����������ƽ�ƾ���
void callback_Translation(int, void*)
{
	Size dst_sz = Img_Translation.size();
	Mat matDst2;

	//����ƽ�ƾ���
	Mat t_mat = Mat::zeros(2, 3, CV_32FC1);

	t_mat.at<float>(0, 0) = 1;
	t_mat.at<float>(0, 2) = dy; //ˮƽƽ����
	t_mat.at<float>(1, 1) = 1;
	t_mat.at<float>(1, 2) = dx; //��ֱƽ����

	//����ƽ�ƾ�����з���任
	warpAffine(Img_Translation, matDst2, t_mat, Size(Img_Translation.cols + dy, Img_Translation.rows + dx));

	//��ʾƽ��Ч��
	imshow("ͼ��ƽ��", matDst2);
	waitKey(0);
}

//ͼ��ƽ��
void Translation() 
{
	//ͼ��ƽ��
	namedWindow("ͼ��ƽ��", WINDOW_AUTOSIZE);
	createTrackbar("X����ƽ���� :", "ͼ��ƽ��", &dx, MAX_DX, callback_Translation);
	createTrackbar("Y����ƽ���� :", "ͼ��ƽ��", &dy, MAX_DY, callback_Translation);
	callback_Translation(0, 0);
	waitKey(0);
}

//ɫ��ƽ��
void equalizeCallback(Mat img)
{
	//ͨ��ʹ��ֱ��ͼ���⻯���������ӶԱȶȣ���������¶���Ȼ�¶�����ϸ��
	Mat result;
	Mat ycrcb;
	//��cvtColor������BGRͼ��ת��ΪYCrCb��ɫ��ʽ
	cvtColor(img, ycrcb, COLOR_BGR2YCrCb);
	//YUV��ͨ������
	vector<Mat> channels;
	split(ycrcb, channels);
	// ��Yͨ�������ȣ�����ֱ��ͼ����
	equalizeHist(channels[0], channels[0]);
	//�ϳ����ɵ�ͨ��������ת��ΪBGR��ʽ
	merge(channels, ycrcb);
	cvtColor(ycrcb, result, COLOR_YCrCb2BGR);
	// �¾ɶԱ�
	imshow("ԭͼ", img);
	imshow("ɫ��ƽ����", result);
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

	//������1���Ҷ����Ա任
	Mat gray = src.clone();
	LinearProc(gray);

	//������2����ֵ�˲�
	salt(src, 3000);//����3000��������255  
	pepper(src, 3000);//����3000��������0  
	// ���뽷��������ͼ��  
	imshow(" ���뽷��������ͼ�� ", src);
	//	���ڴ�С3*3����ֵ�˲�
	medianBlur(src, dst, 3);
	//	��ֵ�˲����ͼ��
	imshow("��ֵ�˲����", dst);
	waitKey(0);

	//������2����ͨ&��ͨ�˲�
	Filter(I);
	HighFilter();
	LowFilter();

	//ѡ����1����ת������
	Img_Rotate=imread("123.jpg");
	Rotate();
	//ѡ����1��ƽ��
	Img_Translation= imread("123.jpg");
	Translation();

	//ѡ����5��ɫ��ƽ��
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
