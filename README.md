# Unsupervised Moving Object Detection

This repo contains the implementation of the method described in the paper

[Unsupervised Moving Object Detection via Contextual Information Separation](https://arxiv.org/pdf/1901.03360.pdf)

Published in the Internation Conference of Computer Vision and Pattern Recognition (CVPR) 2019.

<img src='doc/detect_hawk.gif' width=320>

If you use this code in academic context, please cite the following publication:

```
@inproceedings{yang_loquercio_2019,
  title={Unsupervised Moving Object Detection via Contextual Information Separation},
  author={Yang, Yanchao and Loquercio, Antonio and Scaramuzza, Davide and Soatto, Stefano},
  booktitle = {Conference on Computer Vision and Pattern Recognition {(CVPR)}}
  year={2019}
}
```

Visit the [project webpage](http://rpg.ifi.uzh.ch/unsupervised_detection.html) for more details. For any question, please contact [Antonio Loquercio](https://antonilo.github.io/contact/).

## Running the code

### Prerequisites

This code was tested with the following packages. Note that previous version of them might work but are untested.

* Ubuntu 18.04
* Python3
* Tensorflow 1.13.1
* python-opencv
* CUDA 10.1
* python-gflags
* Keras 2.2.4

### Datasets

We have used three publicly available dataset for our experiments:

[DAVIS 2016](https://davischallenge.org/davis2016/browse.html) | [FBMS59](https://lmb.informatik.uni-freiburg.de/resources/datasets/) | [SegTrackV2](https://web.engr.oregonstate.edu/~lif/SegTrack2/dataset.html)

The datasets can be used without any pre-processing.

### Downloads

### Training

### Testing


### Acknowledgment

Some of the code belonging to this project has been inspired by the following repositories [SfMLearner](https://github.com/tinghuiz/SfMLearner), [TFOptflow](https://github.com/philferriere/tfoptflow), [Generative_Inpaiting](https://github.com/JiahuiYu/generative_inpainting). We would like to thank all the authors of these repositories.

