#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

const char* SOURCE_WINDOW = "Source Image";
const char* WARP_WINDOW = "Warp";
const char* WARP_ROTATE_WINDOW = "Warp + Rotate";
//------
int main(int argc, char** argv)
{
	// Array to store points of affine tranform
	Point2f srcTri[3];
	Point2f dstTri[3];

	// Matrices to tranformation matrices
	Mat rot_mat(2, 3, CV_32FC1), warp_mat(2, 3, CV_32FC1);
	
	// Matrices to store images:
	Mat warp_dst, warp_rotate_dst;

	// Read image 
	Mat src = imread("img1_mod.png", IMREAD_COLOR);


	Rect temp1, temp2;
	int x0 = 478, y0 = 313;
	int x1 = 528, y1 = 357;

	// Define polygon
	// (x0,y0)------(x1,y0)
	//    |            |
	//    |            |
	// (x0,y1)------(x1,y1)
	Point2f Upper_Left_point(x0, y0); 
	Point2f Lower_Left_point(x0, y1);
	Point2f Lower_Right_point(x1, y1);
	Point2f Upper_Right_point(x1, y0);

	Scalar color(255, 0, 0);
	rectangle(src, Upper_Left_point, Lower_Right_point, color, 2, 8, 0);
	imshow("Original Image", src);



	// Define matrix to store output in
	warp_dst = Mat::zeros(src.rows, src.cols, src.type());

	// Define three points to tranform
	srcTri[0] = Point2f(0, 0);
	srcTri[1] = Point2f(src.cols - 1.f, 0);
	srcTri[2] = Point2f(0, src.rows - 1.f);

	// Tranform the three points
	dstTri[0] = Point2f(src.cols*0.0f, src.rows*0.33f);
	dstTri[1] = Point2f(src.cols*0.85f, src.rows*0.25f);
	dstTri[2] = Point2f(src.cols*0.15f, src.rows*0.7f);

	// Calculate tranformation matrix rom the tranformation of the set of points
	warp_mat = getAffineTransform(srcTri, dstTri);
	warpAffine(src, warp_dst, warp_mat, warp_dst.size());

	Point center = Point(warp_dst.cols / 2, warp_dst.rows / 2);
	double angle = -50.0;
	double scale = 0.6;

	rot_mat = getRotationMatrix2D(center, angle, scale);

	warpAffine(warp_dst, warp_rotate_dst, rot_mat, warp_dst.size());
	namedWindow(SOURCE_WINDOW, WINDOW_AUTOSIZE);
	imshow(SOURCE_WINDOW, src);

	namedWindow(WARP_WINDOW, WINDOW_AUTOSIZE);
	imshow(WARP_WINDOW, warp_dst);

	namedWindow(WARP_ROTATE_WINDOW, WINDOW_AUTOSIZE);
	imshow(WARP_ROTATE_WINDOW, warp_rotate_dst);

	waitKey(0);
	return 0;
}
