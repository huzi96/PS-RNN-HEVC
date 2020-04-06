#ifndef H_CAFFEINTRAPRED
#define H_CAFFEINTRAPRED
#include "caffe_head.h"
#include <time.h>

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <memory>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#define USE_OPENCV 1

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;


class HPredict {
public:
  HPredict(const string& model_file, const string& trained_file, bool ifGPU = true);

  void predict(const cv::Mat &img, cv::Mat &dst);

private:

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void WrapOutputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
				  std::vector<cv::Mat>* input_channels);

private:
  std::shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
};

class FCPredict {
public:
  FCPredict(const string& model_file, const string& trained_file, bool ifGPU = true);

  void predict(const std::vector<cv::Mat> &imgs, cv::Mat &dst);

private:

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void WrapOutputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const std::vector<cv::Mat> &imgs, std::vector<cv::Mat>* input_channels);

private:
  std::shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
};


int test_caffe_intra();
int test_caffe_fc();
#endif
