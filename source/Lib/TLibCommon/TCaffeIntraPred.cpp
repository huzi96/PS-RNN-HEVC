// CaffeTest.cpp : Defines the entry point for the console application.
//

#include "TCaffeIntraPred.h"



HPredict::HPredict(const string& model_file, const string& trained_file, bool ifGPU) {


  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
	<< "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

void HPredict::predict(const cv::Mat &img, cv::Mat &dst) {

  Blob<float>* input_layer = net_->input_blobs()[0];

  input_geometry_ = cv::Size(img.cols, img.rows);

  input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  std::vector<cv::Mat> output_channels;
  WrapOutputLayer(&output_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  output_channels[0].copyTo(dst);

}

void HPredict::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
	cv::Mat channel(height, width, CV_32FC1, input_data);
	input_channels->push_back(channel);
	input_data += width * height;
  }
}

void HPredict::WrapOutputLayer(std::vector<cv::Mat>* output_channels) {
  Blob<float>* ourput_layer = net_->output_blobs()[0];

  int width = ourput_layer->width();
  int height = ourput_layer->height();
  float* ouput_data = ourput_layer->mutable_cpu_data();
  for (int i = 0; i < ourput_layer->channels(); ++i) {
	cv::Mat channel(height, width, CV_32FC1, ouput_data);
	output_channels->push_back(channel);
	ouput_data += width * height;
  }
}

void HPredict::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels) {

  cv::Mat sample;
  sample = img;

  cv::Mat sample_resized;
  sample_resized = sample;

  cv::Mat sample_float;
  sample_resized.convertTo(sample_float, CV_32FC1);

  sample_float = sample_float / 255.0;
  cv::split(sample_float, *input_channels);
  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}

int test_caffe_intra() 
{

	//::google::InitGoogleLogging(argv[0]);

  string model_file = "conv_rev3.prototxt";
  string trained_file = "conv_rev3_iter_6500.caffemodel";


  clock_t start_time = clock();
  HPredict hpredict(model_file, trained_file);
  clock_t end_time = clock();

  printf("The running time is: %f\n", static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC);
	
  unsigned short data[64];
  cv::Mat img(24, 24, CV_8UC1, data, 8);
  cv::Mat dst = cv::Mat::zeros(8, 8, CV_32FC1);
  for (int i = 0; i < img.rows; i++)
  {
	uchar* ptr = img.ptr<uchar>(i);
	for (int j = 0; j < img.cols; j++)
	{
	  ptr[j] = 112;
	}
  }
  CHECK(!img.empty()) << "Unable to decode image ";
  hpredict.predict(img, dst);
  return 0;
}


FCPredict::FCPredict(const string& model_file, const string& trained_file, bool ifGPU) {


  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 5)
	<< "Input layer should have 5 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

void FCPredict::predict(const std::vector<cv::Mat> &imgs, cv::Mat &dst) {

  Blob<float>* input_layer = net_->input_blobs()[0];

  input_geometry_ = cv::Size(imgs[0].cols, imgs[0].rows);

  input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  std::vector<cv::Mat> output_channels;
  WrapOutputLayer(&output_channels);

  Preprocess(imgs, &input_channels);

  net_->Forward();

  output_channels[0].copyTo(dst);

}

void FCPredict::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
	cv::Mat channel(height, width, CV_32FC1, input_data);
	input_channels->push_back(channel);
	input_data += width * height;
  }
}

void FCPredict::WrapOutputLayer(std::vector<cv::Mat>* output_channels) {
  Blob<float>* ourput_layer = net_->output_blobs()[0];

  int width = ourput_layer->width();
  int height = ourput_layer->height();
  float* ouput_data = ourput_layer->mutable_cpu_data();
  for (int i = 0; i < ourput_layer->channels(); ++i) {
	cv::Mat channel(height, width, CV_32FC1, ouput_data);
	output_channels->push_back(channel);
	ouput_data += width * height;
  }
}

void FCPredict::Preprocess(const std::vector<cv::Mat> &imgs, std::vector<cv::Mat>* input_channels) {

  cv::Mat sample_float;
  for (int i = 0; i < num_channels_; i++)
  {
	imgs[i].convertTo(sample_float, CV_32FC1);
	sample_float = sample_float / 255.0;
	sample_float.copyTo((*input_channels)[i]);
  }

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}

//
int test_caffe_fc()
{

  string model_file = "mlp_ref1_deploy.prototxt";
  string trained_file = "mlp_ref1_iter_29000.caffemodel";

  clock_t start_time = clock();
  FCPredict fcpredict(model_file, trained_file);
  clock_t end_time = clock();

  printf("The running time is: %f\n", static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC);

  vector<cv::Mat> imgs;
  for (int k = 0; k < 5; k++)
  {
	  imgs.push_back(cv::Mat::zeros(8, 8, CV_32FC1));
	  if (k == 0)
	  for (int i = 0; i < 8; i++)
	  {
		float *ptr = imgs[k].ptr<float>(i);
		for (int j = 0; j < 8; j++)
		{
		  ptr[j] = 255;
		}
	  }
  }
  cv::Mat dst = cv::Mat::zeros(8, 8, CV_32FC1);
  fcpredict.predict(imgs, dst);
  for (int i = 0; i < 8; i++)
  {
	float *ptr = dst.ptr<float>(i);
	for (int j = 0; j < 8; j++)
	{
	  std::cout << ptr[j] << ' ';
	}
	std::cout << std::endl;
  }
  return 0;
}
