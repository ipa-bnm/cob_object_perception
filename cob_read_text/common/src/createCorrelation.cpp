#include <iostream>
#include <fstream>
#include <sstream>
#include <stack>
#include <cstdio>
#include <dirent.h>

// ROS includes
#include <ros/ros.h>
#include <ros/package.h>

// OpenCV includes
#include "opencv2/core/core.hpp"
#include "cv.h"
#include "highgui.h"

int main(int argc, char* argv[])
{
  std::string input_image_path = ros::package::getPath("cob_read_text_data") + "/fonts/letters_arial_600dpi_resized.bmp";
  cv::Mat input = cv::imread(input_image_path.c_str()); //"/home/rmb-rh/Desktop/letters_arial/letters_arial_600dpi_resized.bmp");

  // extract single letter images from big image
  std::vector<cv::Mat> letterImages;
  int x_coord = 66, y_coord = 14;
  for (unsigned int y = 0; y < 16; y++)
  {
    for (unsigned int x = 0; x < 16; x++)
    {
      letterImages.push_back(input(cv::Rect(x_coord, y_coord, 100, 100)));
      x_coord += 228;
    }
    x_coord = 66;
    y_coord += 131;
  }

  // write single letter images as file
  std::string letter_image_folder = ros::package::getPath("cob_read_text_data") + "/fonts/";
  for (unsigned int i = 0; i < letterImages.size(); i++)
  {
    std::string s = letter_image_folder;
    std::stringstream ss;
    ss << i;
    s.append(ss.str());
    s.append(".png");
    std::cout << "s: " << s << std::endl;
    cv::imwrite(s, letterImages[i]);
  }

  // correlate
  std::ofstream correlationFile;
  std::string textname = ros::package::getPath("cob_read_text_data") + "/fonts/correlation.txt";
  correlationFile.open(textname.c_str());

  double maxScore = 0, minScore = 1;
  std::vector<std::vector<double> > scores(letterImages.size());
  for (unsigned int y = 0; y < letterImages.size(); y++)
  {
    scores[y].resize(letterImages.size());
    for (unsigned int x = 0; x < letterImages.size(); x++)
    {
    //  correlationFile << "[" << y << "|" << x << "]: ";
      cv::Mat grayLetter2 = cv::Mat(letterImages[y].size(), CV_8UC1, cv::Scalar(0));
      cv::cvtColor(letterImages[y], grayLetter2, CV_RGB2GRAY);

      cv::Mat grayLetter1 = cv::Mat(letterImages[x].size(), CV_8UC1, cv::Scalar(0));
      cv::cvtColor(letterImages[x], grayLetter1, CV_RGB2GRAY);

      // count difference pixels
      int count = 0;

      for (int yy = 0; yy < grayLetter1.rows; yy++)
        for (int xx = 0; xx < grayLetter1.cols; xx++)
          if ((int)grayLetter1.at<unsigned char> (yy, xx) == (int)grayLetter2.at<unsigned char> (yy, xx))
            count++;

      scores[y][x] = (double)count / (double)(grayLetter1.rows * grayLetter1.cols);

      if (scores[y][x] > maxScore)
    	  maxScore = scores[y][x];
      if (scores[y][x] < minScore)
    	  minScore = scores[y][x];
    }
  }
  // write correlations to file
  double factor = 1.0/(maxScore-minScore);
  for (unsigned int y = 0; y < letterImages.size(); y++)
  {
    for (unsigned int x = 0; x < letterImages.size(); x++)
      correlationFile << (scores[y][x]-minScore)*factor << " ";
    correlationFile << std::endl;
  }

  correlationFile.close();

  return 0;
}
