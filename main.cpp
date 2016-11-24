#include <iostream>
#include <string>

#include <cv.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/stitching/stitcher.hpp"
#include <cvaux.h>

#include "env.h"

using namespace cv;
using namespace std;
typedef struct image
{
    Mat origin;
    Mat gray;
    Mat descriptor;
    vector<KeyPoint> kp;
}image;

SiftFeatureDetector siftdet;
FlannBasedMatcher matcher;
SiftDescriptorExtractor extractor;

Mat Stitched(Mat img1, Mat img2,string str) {
    image I1,I2;
    vector<DMatch> matches, goodmatches;

    //store original image
    I1.origin = img1;
    I2.origin = img2;

    //transfer to gray picture
    cvtColor(I1.origin,I1.gray, CV_BGR2GRAY);
    cvtColor(I2.origin,I2.gray, CV_BGR2GRAY);

    //calculate keypoint
    siftdet.detect(I1.gray, I1.kp);
    siftdet.detect(I2.gray, I2.kp);

    //calculate eigenvector
    extractor.compute(I1.gray, I1.kp, I1.descriptor);
    extractor.compute(I2.gray, I2.kp, I2.descriptor);

    //SIFT match key point
    matcher.match(I1.descriptor, I2.descriptor, matches);
    double max_dist = 0;
    double min_dist = 9999;
    for(int i = 0;i < matches.size();i++)
    {
        if(matches[i].distance < min_dist)
            min_dist = matches[i].distance;
        if(matches[i].distance > max_dist)
            max_dist = matches[i].distance;
    }
    for (int i = 0; i < matches.size(); i++) {
        if (matches[i].distance <  2*max(0.01,min_dist))
            goodmatches.push_back(matches[i]);
    }

    Mat img_matches;
    drawMatches(I1.origin, I1.kp, I2.origin, I2.kp,
                goodmatches, img_matches, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("matches" + str, img_matches);

    int numRsideKp1 = 0;
    int numRsideKp2 = 0;
    for(auto v:goodmatches)
    {
        if(I1.kp[v.queryIdx].pt.x > I1.origin.cols /2)
            numRsideKp1++;
        if(I2.kp[v.trainIdx].pt.x > I2.origin.cols /2)
            numRsideKp2++;
    }
    Mat imgL,imgR;
    vector<Point2f> Lkey, Rkey;
    if(numRsideKp1 > numRsideKp2)
    {
        imgL = I1.origin.clone();
        imgR = I2.origin.clone();
        for(auto v :goodmatches)
        {
            Lkey.push_back(I1.kp[v.queryIdx].pt);
            Rkey.push_back(I2.kp[v.trainIdx].pt);
        }
    }
    else
    {
        imgL = I2.origin.clone();
        imgR = I1.origin.clone();
        for(auto v :goodmatches)
        {
            Lkey.push_back(I2.kp[v.trainIdx].pt);
            Rkey.push_back(I1.kp[v.queryIdx].pt);
        }
    }
    //Transfer Matrix calculate
    Mat H = findHomography(Rkey, Lkey, CV_RANSAC);
    int rows = imgL.rows > imgR.rows? imgL.rows:imgR.rows;

    Mat stitchedImage;
    warpPerspective(imgR, stitchedImage, H, Size((imgL.cols + imgR.cols) * 2.5, imgL.cols + imgR.cols),INTER_LINEAR);
    imshow("half",stitchedImage);
    Mat L(stitchedImage, Rect(0, 0, imgL.cols, imgL.rows));
    imgL.copyTo(L);
    Mat stitchedgray;

    cvtColor(stitchedImage,stitchedgray,CV_BGR2GRAY);
    medianBlur(stitchedgray,stitchedgray,3);
    threshold(stitchedgray,stitchedgray,1,255,0);

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(stitchedgray,contours,hierarchy,CV_RETR_LIST,CV_CHAIN_APPROX_SIMPLE);
    int max_area = 0;
    Rect max_rect ;
    for(auto c: contours)
    {
        Rect rect = boundingRect(c);
        auto area = rect.height *rect.width;
        if(area > max_area)
        {
            max_area = area;
            max_rect = rect;
        }
    }
    Mat roiImg = stitchedImage(max_rect).clone();
    return roiImg;
}
int main() {
    Mat stImg;
    Mat img1 = imread(PATH[0],CV_LOAD_IMAGE_UNCHANGED);

    resize(img1, img1, Size(img1.cols / scale_coefficient, img1.rows / scale_coefficient));
    imshow("i",img1);
    for(int i = 1;i < PATH.size();i++)
    {
        Mat img2 = imread(PATH[i],CV_LOAD_IMAGE_UNCHANGED);
        resize(img2, img2, Size(img2.cols / scale_coefficient, img2.rows / scale_coefficient));
        stringstream out;
        out << i;
        string str = out.str();
        stImg = Stitched(img1,img2,str);
        img1 = stImg;
    }

    imshow("result",stImg);
    imwrite("/Users/haoranzhi/Desktop/out.JPG",stImg);
    while(1)
        waitKey(0);
    return 0;
}
