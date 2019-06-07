### Unsupervised moving object segmentation

This code implements the adversarial idea of having a motion segmentation
without supervision. The master branch contains clean code which is used in
the code release.

In case you want to check old code for pretraining the recover with box masking
or check out some old functions (for example the data augmentation ones) please
check the branch "triple_gen"


## Dependencies

This library has the following dependencies:

0. Python (3.n versions are better)
1. [Tensorflow](https://www.tensorflow.org/install/)
2. Keras, ```sudo pip install keras```
3. numpy, ```sudo pip install numpy```
4. gflags, ```sudo pip install python-gflags ```
5. PWCNET (included model checkpoint)
6. Pre-trained model for recover.

