#include <caffe/common.hpp>
#include <caffe/layer.hpp>
#include <caffe/layer.hpp>  
#include <caffe/layer_factory.hpp>  
#include <caffe/layers/input_layer.hpp>  
#include <caffe/layers/inner_product_layer.hpp>  
#include <caffe/layers/dropout_layer.hpp>  
#include <caffe/layers/conv_layer.hpp>  
#include <caffe/layers/relu_layer.hpp> 
#include <caffe/layers/concat_layer.hpp> 
#include <caffe/layers/prelu_layer.hpp>  
#include <caffe/layers/pooling_layer.hpp>  
#include <caffe/layers/lrn_layer.hpp>  
#include <caffe/layers/softmax_layer.hpp> 
#include <caffe/layers/silence_layer.hpp> 
#include <caffe/layers/slice_layer.hpp> 
#include <caffe/layers/flatten_layer.hpp>
#include <caffe/layers/reshape_layer.hpp>
#include <caffe/layers/deconv_layer.hpp>
#include <caffe/layers/crop_layer.hpp>
#include <caffe/layers/eltwise_layer.hpp>
#include <caffe/vision_layers.hpp>
#include <caffe/layers/im2col_layer.hpp>

namespace caffe
{
#ifndef INSTANTIATED
#define INSTANTIATED
	extern INSTANTIATE_CLASS(InputLayer);
	extern INSTANTIATE_CLASS(InnerProductLayer);
	extern INSTANTIATE_CLASS(DropoutLayer);
	extern INSTANTIATE_CLASS(ConvolutionLayer);
	REGISTER_LAYER_CLASS(Convolution);
	extern INSTANTIATE_CLASS(ReLULayer);
	REGISTER_LAYER_CLASS(ReLU);
	extern INSTANTIATE_CLASS(PoolingLayer);
	REGISTER_LAYER_CLASS(Pooling);
	extern INSTANTIATE_CLASS(LRNLayer);
	REGISTER_LAYER_CLASS(LRN);
	extern INSTANTIATE_CLASS(SoftmaxLayer);
	REGISTER_LAYER_CLASS(Softmax);
	extern INSTANTIATE_CLASS(PReLULayer);
	REGISTER_LAYER_CLASS(PReLU);
	extern INSTANTIATE_CLASS(ConcatLayer);
	REGISTER_LAYER_CLASS(Concat);
	extern INSTANTIATE_CLASS(SilenceLayer);
	REGISTER_LAYER_CLASS(Silence);
	extern INSTANTIATE_CLASS(SliceLayer);
	REGISTER_LAYER_CLASS(Slice);
	extern INSTANTIATE_CLASS(FlattenLayer);
	REGISTER_LAYER_CLASS(Flatten);
	extern INSTANTIATE_CLASS(ReshapeLayer);
	REGISTER_LAYER_CLASS(Reshape);
	extern INSTANTIATE_CLASS(DeconvolutionLayer);
	REGISTER_LAYER_CLASS(Deconvolution);

	extern INSTANTIATE_CLASS(CropLayer);
	REGISTER_LAYER_CLASS(Crop);

	extern INSTANTIATE_CLASS(EltwiseLayer);
	REGISTER_LAYER_CLASS(Eltwise);

	extern INSTANTIATE_CLASS(ResizeLayer);
	REGISTER_LAYER_CLASS(Resize);

	extern INSTANTIATE_CLASS(GateRecurrentLayer);
	REGISTER_LAYER_CLASS(GateRecurrent);

	extern INSTANTIATE_CLASS(GateLstmLayer);
	REGISTER_LAYER_CLASS(GateLstm);

	extern INSTANTIATE_CLASS(RegionconvolutionLayer);
	REGISTER_LAYER_CLASS(Regionconvolution);
#endif

}

//
//#include <caffe/caffe.hpp>
//#include <caffe/common.hpp>
//#include <caffe/layer.hpp>  
//#include <caffe/layer_factory.hpp>  
//#include <caffe/layers/input_layer.hpp>  
//#include <caffe/layers/inner_product_layer.hpp>  
//#include <caffe/layers/dropout_layer.hpp>  
//#include <caffe/layers/conv_layer.hpp>  
//#include <caffe/layers/relu_layer.hpp> 
//#include <caffe/layers/concat_layer.hpp> 
//#include <caffe/layers/prelu_layer.hpp>  
//#include <caffe/layers/pooling_layer.hpp>  
//#include <caffe/layers/lrn_layer.hpp>  
//#include <caffe/layers/softmax_layer.hpp>   
//
//#include "stdafx.h"
//#include <time.h>
//
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//
//#include <algorithm>
//#include <iosfwd>
//#include <memory>
//#include <string>
//#include <utility>
//#include <vector>
//
//
//	namespace caffe
//	{
//
//		extern INSTANTIATE_CLASS(InputLayer);
//		extern INSTANTIATE_CLASS(InnerProductLayer);
//		extern INSTANTIATE_CLASS(DropoutLayer);
//		extern INSTANTIATE_CLASS(ConvolutionLayer);
//		REGISTER_LAYER_CLASS(Convolution);
//		extern INSTANTIATE_CLASS(ReLULayer);
//		REGISTER_LAYER_CLASS(ReLU);
//		extern INSTANTIATE_CLASS(PoolingLayer);
//		REGISTER_LAYER_CLASS(Pooling);
//		extern INSTANTIATE_CLASS(LRNLayer);
//		REGISTER_LAYER_CLASS(LRN);
//		extern INSTANTIATE_CLASS(SoftmaxLayer);
//		REGISTER_LAYER_CLASS(Softmax);
//		extern INSTANTIATE_CLASS(PReLULayer);
//		REGISTER_LAYER_CLASS(PReLU);
//		extern INSTANTIATE_CLASS(ConcatLayer);
//		REGISTER_LAYER_CLASS(Concat);
//	}
//
//
//	using namespace caffe;  // NOLINT(build/namespaces)
//	using std::string;
//
//#define USE_OPENCV 1
//
//
//
//	/* Pair (label, confidence) representing a prediction. */
//	typedef std::pair<string, float> Prediction;
//
//	class Interpolator {
//	public:
//		Interpolator(const string& model_file,
//			const string& trained_file);
//
//		std::vector<cv::Mat>  interpolate(const cv::Mat& img);
//
//	private:
//
//		void WrapInputLayer(std::vector<cv::Mat>* input_channels);
//
//		void WrapOutputLayer(std::vector<cv::Mat>* input_channels);
//
//		void Preprocess(const cv::Mat& img,
//			std::vector<cv::Mat>* input_channels);
//
//	private:
//		std::shared_ptr<Net<float> > net_;
//		cv::Size input_geometry_;
//		int num_channels_;
//		cv::Mat mean_;
//		std::vector<string> labels_;
//	};