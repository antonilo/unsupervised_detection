# Unsupervised Moving Object Detection

This repo contains the implementation of the method described in the paper

[Unsupervised Moving Object Detection via Contextual Information Separation](https://arxiv.org/pdf/1901.03360.pdf)

Published in the International Conference of Computer Vision and Pattern Recognition (CVPR) 2019.

For a brief overview, check out the project [VIDEO](https://youtu.be/01vClielQBw)!

<img src='doc/detect_hawk.gif' width=380>

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

We generate optical flows with a tensorflow implementation of PWCNet, which is an adapted version of [this repository](https://github.com/philferriere/tfoptflow).
To compute flows, please download the model checkpoint of PWCNet we used for our experiments, available at [this link](https://drive.google.com/open?id=1gtGx_6MjUQC5lZpl6-Ia718Y_0pvcYou).

Additionally, you can find our trained models in the [project webpage](http://rpg.ifi.uzh.ch/unsupervised_detection.html).

### Training

Once you have downloaded the datasets (at least one of the three), you can start training the model.
All the required flags (and their defaults) are explained in the [common\_flags.py](./common_flags.py) file.

The folder [scripts](./scripts) contains an example of how to train a model on the DAVIS dataset.
To start training, edit the file [train\_DAVIS2016.sh](./scripts/train_DAVIS2016.sh) and add there the paths to the dataset and to the PWCNet checkpoint. After that you should be able to start training with the following command:
```bash
bash ./scripts/train_DAVIS2016.sh
```

You can monitor the training process in `tensorboard` running the following command
```bash
tensorboard --logdir=/path/to/tensorflow/log/files
```
and by opening [https://localhost:6006](https://localhost:6006) on your browser.

To speed up training, we pre-trained the recover on the task of optical flow in-painting on box-shaped occlusions.
We used the [Flying Chair](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html) dataset for this training.
The resulting checkpoint is used to initialize the recover network before the adversarial training.
This checkpoint can be found together with our trained models in the [project webpage](http://rpg.ifi.uzh.ch/unsupervised_detection.html).
Although not strictly required, the recover pre-training significantly speeds up model convergence.

### Testing

You can test a trained model with the function [test\_generator.py](./test_generator.py).
An example is provided for the DAVIS 2016 dataset in the [scripts](./scripts) folder.
To run it, edit the file [test\_DAVIS2016\_raw.sh](./scripts/test_DAVIS2016_raw.sh) with the paths to the dataset, the optical flow and the model checkpoint. After that, you can test the model with the following command:
```bash
bash ./scripts/test_DAVIS2016_raw.sh
```

### Post-Processing

Raw predictions are post-processed to increase model accuracy. In particular, the post-processing is composed of two steps: (i) averaging the predictions over different time shifts between the first and second image, as well as for multiple central crops, and (ii) Conditional Random Fields (CRF) of the average predictions and best candidate mask selection. 

To generate predictions over multiple time steps and crops for the DAVIS 2016 dataset, please use the [generate\_buffer\_DAVIS2016.sh](./scripts/generate_buffer_DAVIS2016.sh) script. This can be done by editing the script to add the path to the dataset, the PWCNet and the trained model checkpoints. 

After predictions buffers are generated, please use the [post-processing script](./post_processing/post_processing.py) to compute refined predictions.

### Pre-Computed Results

Our final, post-processed results are available for the DAVIS 2016, FBMS59 and SegTrackv2 datasets at [this link](http://rpg.ifi.uzh.ch/data/detection_results.zip). In case you will evaluate on other datasets and would like to share the predictions please contact us!

### FAQ

_The training loss seems symmetric for the mask and its complementary. How do you tell which one is the foreground and which the background?_

For the training process it is very important to keep this symmetry. Without it the optimum of the training process is not guaranteed to be the separation of independent components anymore. However, to detect whether the masks cover the object or the background, we use the heuristic that background usually occupies more than two boundaries of the image. You can find the corresponding implementation of this heuristic in the function [disambiguate_forw_back](models/utils/general_utils.py#L100).


### Acknowledgment

Some of the code belonging to this project has been inspired by the following repositories [SfMLearner](https://github.com/tinghuiz/SfMLearner), [TFOptflow](https://github.com/philferriere/tfoptflow), [Generative_Inpaiting](https://github.com/JiahuiYu/generative_inpainting). We would like to thank all the authors of these repositories.
