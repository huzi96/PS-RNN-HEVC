# Progressive Spatial Recurrent Neural Network for Intra Prediction
## Overview
This is the implementation of ther paper,
> Yueyu Hu, Wenhan Yang, Mading Li, and Jiaying Liu, 
> Progressive Spatial Recurrent Neural Network for Intra Prediction,
> <i>IEEE Transactions on MultiMedia</i> (<i>TMM</i>), 2019


The paper is also available at <url>https://arxiv.org/abs/1807.02232</url>.

The implementation is based on HEVC Test Model (HM) 16.15. We implement PS-RNN models using TensorFlow Library (v1.12), and the model is integrated into the codec using libtensorflow (C API).

## Build
We currently support building on Linux. To automatically build the executables, please first enter ```build/linux ```. In that directory, run

``` CPLUS_INCLUDE_PATH=$PWD/../../extern/include:$CPLUS_INCLUDE_PATH make```

to generate the executables. After that, in the same directory, you may export the shared library path of libtensorflow into ```LD_LIBRARY_PATH``` by executing

```export LD_LIBRARY_PATH=$PWD/../../extern/lib:$LD_LIBRARY_PATH```

All paths can be modified to customize for your own environment.

We compile libtensorflow in CPU-only mode, with AVX2 acceleration enabled. You can also download or build other libtensorflow binaries to the ```extern``` folder. Note that different versions of libraries may vary in behavior and sometimes crashes the program because of incompatible memory management.

## Run
We provide an example in the ```tests``` folder. Encode the first frame of the provided ```blowingbubble``` sequence, with QP=37, by doing

```../bin/TAppEncoderStatic -c encoder_intra_main_4-32.cfg -c ./per-sequence/BlowingBubbles.cfg -i BlowingBubbles_416x240_50.yuv -q  37 -f 1 -b BlowingBubbles_416x240_50_qp37.bin -o BlowingBubbles_416x240_50_qp37.yuv```

in the ```tests``` directory. And you can run the decoding by executing

```../bin/TAppDecoderStatic -b ./BlowingBubbles_416x240_50_qp37.bin -o BlowingBubbles_416x240_50_qp37_dec.yuv```

Other sequences can be tested in similar ways.

## Notes
This is an initial release of the code. We provide the model description and the script to make TensorFlow models into ```.pb``` files in the ```util``` directory as the examples, but the auxiliary files processed by the scripts are currently not available. Documents and comments for the code are also to be added in future updates.