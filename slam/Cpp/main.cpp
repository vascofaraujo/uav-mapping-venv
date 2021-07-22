#include <iostream>
#include <unistd.h>
#include <string>
#include <filesystem>
#include <fstream>
#include <vector>

#include <stack>
#include <ctime>
#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

std::stack<clock_t> tictoc_stack;

void tic() {
    tictoc_stack.push(clock());
}

void toc() {
    std::cout << "Time elapsed: "
              << ((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC
              << std::endl;
    tictoc_stack.pop();
}

int ratioTest(std::vector<std::vector<cv::DMatch>>& matches) {

	int removed=0;
    int ratio = 0.65f;
    int aux=0;
    // for all matches
	for (std::vector<std::vector<cv::DMatch>>::iterator matchIterator= matches.begin();
		 matchIterator!= matches.end(); ++matchIterator) {

		 // if 2 NN has been identified
		 if (matchIterator->size() > 1) {

			 // check distance ratio
			 if ((*matchIterator)[0].distance/(*matchIterator)[1].distance > ratio) {

				 matchIterator->clear(); // remove match
				 removed++;
			 }
             aux++;

		 } else { // does not have 2 neighbours

			 matchIterator->clear(); // remove match
			 removed++;
		 }
	}
    std::cout << "aux"<<aux;
	return removed;
}

void match_points(cv::Mat img1, cv::Mat img2){
    cv::Ptr<cv::SIFT> detector =cv::SIFT::create();
    //cv::Ptr<cv::ORB> detector = cv::ORB::create();
    //cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    img1.convertTo(img1, CV_8U);
    img2.convertTo(img2, CV_8U);
    	
    //cv::cvtColor(img1, img1, cv::COLOR_BGR2GRAY);
    //cv::cvtColor(img2, img2, cv::COLOR_BGR2GRAY);

    tic();
    detector->detectAndCompute( img1, cv::noArray(), keypoints1, descriptors1 );
    detector->detectAndCompute( img2, cv::noArray(), keypoints2, descriptors2 );
    
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector< std::vector< cv::DMatch >> matches;
    matcher.knnMatch(descriptors1, descriptors2,matches, 2); 
    toc();


    int removed= ratioTest(matches);
    std::cout << "removed "<< removed;


    std::cout << matches.size() << '\n';
    std::cout << matches[0].size() << '\n';
    /*cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch> matches;

    matcher.match(descriptors1,descriptors2, matches,2);*/
    

    std::vector<int> queryIdxs( matches.size() ), trainIdxs( matches.size() );
    
    for( size_t i = 0; i < matches.size(); i++ ){
        queryIdxs[i] = matches[i][0].queryIdx;
        trainIdxs[i] = matches[i][0].trainIdx;
    }

    std::vector<cv::Point2f> points1; 
    cv::KeyPoint::convert(keypoints1, points1, queryIdxs);
    std::vector<cv::Point2f> points2; 
    cv::KeyPoint::convert(keypoints2, points2, trainIdxs);
    
    int ransacThresh=5;
    cv::Mat H12;
    cv::Mat mask;
    H12 = cv::findHomography( cv::Mat(points2), cv::Mat(points1), cv::RANSAC, ransacThresh , mask);
    
    cv::Mat img2Out;
    //cv::warpPerspective(img2, img2Out, H12, cv::Size(img2.cols, img2.rows), cv::INTER_LINEAR);
    //cv::drawKeypoints(img1, keypoints1, img2Out, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches[1], img2Out, cv::Scalar::all(-1), cv::Scalar(0,0,255),std::vector<char>() );

    cv::imshow("blablabla", img2Out);
    int k = cv::waitKey(0);
}






int focalLength = 525.0;
int centerX = 319.5;
int centerY = 239.5;
int scalingFactor = 5000.0;

int main (void){
    
    
    char tmp[256];
    getcwd(tmp, 256);
    std::cout << "Current directory " << tmp << std::endl;
    std::ifstream inFile;
    inFile.open("/home/renato/Documents/git-repo/Point_cloud/sync.txt");
    std::vector<std::vector<std::string>> mat;
    std::string a;
    std::string b;
    
    //char output[100];
    while (!inFile.eof()){
        inFile >> a; // auxiliar
        inFile >> b;
        mat.push_back({a,b});
        //std::cout << a;
        //fflush(stdout);
    }
    inFile.close();

    
    char filename1[]="1311878262.166343.png";
    char filename2[]="1311878262.200042.png";
    
    cv::Mat image1   = cv::imread(filename1, cv::IMREAD_ANYCOLOR);
    cv::Mat image2   = cv::imread(filename2, cv::IMREAD_ANYCOLOR);
    
    cv::imshow("bla", image1);
    int k = cv::waitKey(0);
    
   
    match_points(image1, image2);
    
    
    
    return EXIT_SUCCESS;
}
