#include <vector>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/ocl.hpp>

using std::cout;
using std::cerr;
using std::vector;
using std::string;

using namespace cv;
using cv::FastFeatureDetector;
using cv::SimpleBlobDetector;

using cv::DMatch;
using cv::BFMatcher;
using cv::DrawMatchesFlags;
using cv::Feature2D;
using cv::ORB;
using cv::BRISK;
using cv::AKAZE;
using cv::KAZE;

using cv::xfeatures2d::BriefDescriptorExtractor;
using cv::xfeatures2d::SURF;
using cv::xfeatures2d::SIFT;
using cv::xfeatures2d::DAISY;
using cv::xfeatures2d::FREAK;

const double kDistanceCoef = 4.0;
const int kMaxMatchingSize = 50;
string desc_matcher("bf");
//----------------------------
inline void detect_and_compute(string type, cv::Mat& img, vector<cv::KeyPoint>& kpts, cv::Mat& desc) {
    if (type.find("fast") == 0) {
        type = type.substr(4);
        cv::Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(10, true);
        detector->detect(img, kpts);
    }
    if (type.find("blob") == 0) {
        type = type.substr(4);
        cv::Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create();
        detector->detect(img, kpts);
    }
    if (type == "surf") {
        cv::Ptr<Feature2D> surf = SURF::create(800.0);
        surf->detectAndCompute(img, cv::Mat(), kpts, desc);
    }
    if (type == "sift") {
        cv::Ptr<Feature2D> sift = SIFT::create();
        sift->detectAndCompute(img, cv::Mat(), kpts, desc);
    }
    if (type == "orb") {
        cv::Ptr<ORB> orb = ORB::create();
        orb->detectAndCompute(img, cv::Mat(), kpts, desc);
    }
    if (type == "brisk") {
        cv::Ptr<BRISK> brisk = BRISK::create();
        brisk->detectAndCompute(img, cv::Mat(), kpts, desc);
    }
    if (type == "kaze") {
        cv::Ptr<KAZE> kaze = KAZE::create();
        kaze->detectAndCompute(img, cv::Mat(), kpts, desc);
    }
    if (type == "akaze") {
        cv::Ptr<AKAZE> akaze = AKAZE::create();
        akaze->detectAndCompute(img, cv::Mat(), kpts, desc);
    }
    if (type == "freak") {
        cv::Ptr<FREAK> freak = FREAK::create();
        freak->compute(img, kpts, desc);
    }
    if (type == "daisy") {
        cv::Ptr<DAISY> daisy = DAISY::create();
        daisy->compute(img, kpts, desc);
    }
    if (type == "brief") {
        cv::Ptr<BriefDescriptorExtractor> brief = BriefDescriptorExtractor::create(64);
        brief->compute(img, kpts, desc);
    }
}
//---------------
inline void match(string type, cv::Mat& desc1, cv::Mat& desc2, vector<DMatch>& matches) {
    matches.clear();
    if (type == "bf") {
        BFMatcher desc_matcher(cv::NORM_L2, false);
        desc_matcher.match(desc1, desc2, matches, cv::Mat());
    }
    if (type == "knn") {
        BFMatcher desc_matcher(cv::NORM_L2, true);
        vector< vector<DMatch> > vmatches;
        desc_matcher.knnMatch(desc1, desc2, vmatches, 1);
        for (int i = 0; i < static_cast<int>(vmatches.size()); ++i) {
            if (!vmatches[i].size()) {
                continue;
            }
            matches.push_back(vmatches[i][0]);
        }
    }
    std::sort(matches.begin(), matches.end());
    while (matches.front().distance * kDistanceCoef < matches.back().distance) {
        matches.pop_back();
    }
    while (matches.size() > kMaxMatchingSize) {
        matches.pop_back();
    }
}
//---------------------------------
inline void findKeyPointsHomography(vector<cv::KeyPoint>& kpts1, vector<cv::KeyPoint>& kpts2,
        vector<DMatch>& matches, vector<char>& match_mask) {
    if (static_cast<int>(match_mask.size()) < 3) {
        return;
    }
    vector<cv::Point2f> pts1;
    vector<cv::Point2f> pts2;
    for (int i = 0; i < static_cast<int>(matches.size()); ++i) {
        pts1.push_back(kpts1[matches[i].queryIdx].pt);
        pts2.push_back(kpts2[matches[i].trainIdx].pt);
    }
    findHomography(pts1, pts2, cv::RANSAC, 4, match_mask);
}
//------
int main(int argc, char** argv) {

    string img_file1("image.jpeg");
    string img_file2("image.jpeg");

    cv::Mat img1 = cv::imread(img_file1, CV_LOAD_IMAGE_COLOR);
    cv::Mat img2 = cv::imread(img_file2, CV_LOAD_IMAGE_COLOR);

    if (img1.channels() != 1)
    	cvtColor(img1, img1, cv::COLOR_RGB2GRAY);
    if (img2.channels() != 1)
        cvtColor(img2, img2, cv::COLOR_RGB2GRAY);

    vector<cv::KeyPoint> kpts1, kpts2;
    cv::Mat desc1, desc2;
    detect_and_compute("surf", img1, kpts1, desc1);
    detect_and_compute("surf", img2, kpts2, desc2);

    vector<DMatch> matches;
    match("bf", desc1, desc2, matches);

    vector<char> match_mask(matches.size(), 1);
    findKeyPointsHomography(kpts1, kpts2, matches, match_mask);

    cv::Mat res;
    cv::drawMatches(img1, kpts1, img2, kpts2, matches, res, cv::Scalar::all(-1),
        cv::Scalar::all(-1), match_mask, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imshow("result", res);


    cv::waitKey(0);
    return 0;
}
