"""
This File implements the data reader for the SegTrackV2
Dataset. See the file davis2016_data_utils.py for
a more detailed documentation of the functions.
"""
import numpy as np
import os
import tensorflow as tf
from data.aug_flips import random_flip_images

class DirectoryIterator(object):
    """
    Class for managing data loading.of images and labels
    We assume that the folder structure is:

    """
    def __init__(self, directory):
        self.directory = directory

        all_files = os.path.join(directory, 'ImageSets/all.txt')
        self.image_dirs = os.path.join(self.directory, 'JPEGImages')
        self.annotation_dir = os.path.join(self.directory, 'GroundTruth')
        if not os.path.isfile(all_files):
            raise IOError("Division file not found")
        self.components = np.loadtxt(all_files, dtype=np.str)
        self.components = [c[1:] for c in self.components]

        # First count how many experiments are out there
        self.samples = 0
        self.num_experiments = 0
        # Each filename is a tuple image / components
        self.image_filenames = []
        self.annotation_filenames = []
        for experiment in self.components:
            self._parse_experiment(experiment)
            self.num_experiments += 1


        if self.samples == 0:
            raise IOError("Did not find any file in the dataset folder")
        assert self.num_experiments == len(self.image_filenames), "Reading failed"

        print('Found {} images belonging to {} experiments.'.format(
                self.samples, self.num_experiments))

    def _parse_experiment(self, experiment):
        """
        Read the files belonging to a  directory
        """
        current_filenames = []
        current_annotations = []
        experiment_file = os.path.join(self.directory, 'ImageSets',
                                        experiment + '.txt')
        assert os.path.isfile(experiment_file), "Experiment {} not found".format(experiment_file)

        all_exp_fnames = np.loadtxt(experiment_file, dtype=np.str, skiprows=1)

        for exp_fname in all_exp_fnames:
            current_filenames.append(os.path.join(self.image_dirs,
                                                  experiment, exp_fname + '.png'))
            assert os.path.isfile(current_filenames[-1]), \
                "Not found image {}".format(current_filenames[-1])
            current_annotations.append(os.path.join(self.annotation_dir,
                                                    experiment, exp_fname + '.png'))
            assert os.path.isfile(current_annotations[-1]), \
                "Not found image {}".format(current_annotations[-1])
            self.samples += 1
        # Append the last
        self.image_filenames.append(current_filenames)
        self.annotation_filenames.append(current_annotations)


class SegTrackV2Reader(object):
    def __init__(self, root_dir, max_temporal_len=3, min_temporal_len=2,
                 num_threads=6):
        self.root_dir = root_dir
        self.max_temporal_len = max_temporal_len
        self.min_temporal_len = min_temporal_len
        self.num_threads = num_threads

    def get_filenames_list(self):
        iterator = DirectoryIterator(self.root_dir)
        filenames, annotation_filenames = iterator.image_filenames, \
                                            iterator.annotation_filenames
        #Training calls it before, so it will be overwritten
        self.val_samples = iterator.samples
        return filenames, annotation_filenames

    def preprocess_image(self, img):
        orig_width = 640
        orig_height = 384
        img = ( tf.cast(img,tf.float32) / tf.constant(255.0) ) - 0.5
        img = tf.image.resize_images(img, [orig_height, orig_width])
        return img

    def preprocess_mask(self, mask):
        orig_width = 640
        orig_height = 384
        mask = (tf.cast(mask,tf.float32) / tf.constant(255.0))
        mask = tf.image.resize_images(mask, [orig_height, orig_width],
                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return mask

    def random_crop_image_pair(self, image_1, image_2, max_cropping_percent=0.9):
        '''
        Produces an (equal) random crop for image_1 and image_2 that is
        at minimum max_cropping_percent smaller than the original image.
        The resulting patch is then reshaped to original size
        '''
        rand = tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)
        cropping_percent = max_cropping_percent + rand*(1-max_cropping_percent)

        image_width = image_1.get_shape().as_list()[1]
        image_height = image_1.get_shape().as_list()[0]
        num_channels = image_1.get_shape().as_list()[2]

        crop_width = tf.cast(image_width*cropping_percent, tf.int32)
        crop_height = tf.cast(image_height*cropping_percent, tf.int32)
        image_c = tf.concat((image_1, image_2), axis=-1)
        image_c = tf.random_crop(image_c, size=[crop_height,
                                                crop_width,
                                                num_channels*2])
        image_c.set_shape([None, None, num_channels*2])
        # Resize
        image_c = tf.image.resize_images(image_c,
                                         [image_height, image_width])
        image_1 = image_c[:,:,:3]
        image_2 = image_c[:,:,3:6]

        return image_1, image_2

    def central_cropping(self, img, cropping_percent):
        orig_height, orig_width = img.get_shape().as_list()[0:2]
        img = tf.image.central_crop(img, cropping_percent)
        img = tf.image.resize_images(img, [orig_height, orig_width])
        return img

    def augment_pair(self, image_1, image_2):
        # Random flips
        image_1, image_2 = random_flip_images(image_1, image_2)
        image_1, image_2 = self.random_crop_image_pair(image_1, image_2,
                                                       self.train_crop)

        return image_1, image_2

    def dataset_map(self, input_queue):
        fname_number, direction = input_queue[0], input_queue[1]
        # Take care with the casting when sampling!!
        t_shift = tf.random_uniform(shape=[], minval=self.min_temporal_len,
                                    maxval=self.max_temporal_len+1,
                                    dtype=tf.int32)
        t_shift = tf.cast(t_shift, dtype=tf.float32)
        img2_fname_number = t_shift * direction + fname_number
        # Conversions
        fname_number = tf.cast(fname_number, dtype=tf.int32)
        img2_fname_number = tf.cast(img2_fname_number, dtype=tf.int32)
        # Reading
        fname_1 = tf.gather(self.filenames, fname_number)
        fname_2 = tf.gather(self.filenames, img2_fname_number)
        file_content = tf.read_file(fname_1)
        image_1 = tf.image.decode_jpeg(file_content, channels=3)
        image_1 = self.preprocess_image(image_1)
        file_content = tf.read_file(fname_2)
        image_2 = tf.image.decode_jpeg(file_content, channels=3)
        image_2 = self.preprocess_image(image_2)
        # Data augmentation
        image_1, image_2 = self.augment_pair(image_1, image_2)

        return image_1, image_2

    def image_inputs(self, batch_size=32, train_crop=1.0,
                     num_threads=6):
        # Reads input data in batches
        t_len = self.max_temporal_len
        file_list, _ = self.get_filenames_list()
        self.train_crop = train_crop
        # Accumulates subsequent filenames, and makes a dataset with
        # end-points.
        N = 0
        last_fname_numbers = [] # Will be used to calculate flow backward
        first_fname_numbers = [] # Will be used to calculate flow forward
        for fnames in file_list:
            last_fname_numbers.append(np.arange(N + t_len, N + len(fnames),
                                dtype=np.int32))
            first_fname_numbers.append(np.arange(N, N + len(fnames) - t_len,
                                dtype=np.int32))
            N += len(fnames)

        self.filenames = np.concatenate(file_list)
        last_fname_numbers = np.concatenate(last_fname_numbers)
        last_fname_numbers = np.vstack((last_fname_numbers, -1.0*np.ones_like(last_fname_numbers))).T

        first_fname_numbers = np.concatenate(first_fname_numbers)
        first_fname_numbers = np.vstack((first_fname_numbers, 1.0*np.ones_like(first_fname_numbers))).T
        all_fname_numbers = np.vstack((first_fname_numbers, last_fname_numbers))
        all_fname_numbers = np.asarray(all_fname_numbers, dtype=np.float32)
        np.random.shuffle(all_fname_numbers)

        # Form training batches
        dataset = tf.data.Dataset.from_tensor_slices(all_fname_numbers)
        dataset = dataset.shuffle(buffer_size=all_fname_numbers.shape[0],
                                  reshuffle_each_iteration=True)
        dataset = dataset.repeat(None)
        dataset = dataset.map(self.dataset_map,
                              num_parallel_calls=self.num_threads)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        dataset = dataset.prefetch(buffer_size=3*batch_size)
        iterator = dataset.make_initializable_iterator()
        img1s, img2s = iterator.get_next()

        return (img1s, img2s, tf.constant(1.0)), iterator


    def test_inputs(self, batch_size=32, t_len=2, with_fname=False,
                    test_crop=1.0):
        # Reads test data in batches
        file_list, annotation_fnames_list = self.get_filenames_list()
        self.test_crop = test_crop
        # Accumulates subsequent filenames, and makes a dataset with
        # end-points.
        N = 0
        last_fname_numbers = [] # Will be used to calculate flow backward
        first_fname_numbers = [] # Will be used to calculate flow forward for the first frames
        for fnames in file_list:
            if t_len < 0:
                last_fname_numbers.append(np.arange(N + abs(t_len), N + len(fnames),
                                    dtype=np.int32))
                first_fname_numbers.append(np.arange(N, N + abs(t_len),
                                    dtype=np.int32))
            elif t_len > 0:
                first_fname_numbers.append(np.arange(N, N + len(fnames) - t_len,
                                    dtype=np.int32))
                last_fname_numbers.append(np.arange(N + len(fnames) - t_len, N + len(fnames),
                                    dtype=np.int32))
            N += len(fnames)

        self.test_t_len = abs(t_len)
        self.filenames = np.concatenate(file_list)
        self.annotation_filenames = np.concatenate(annotation_fnames_list)
        last_fname_numbers = np.concatenate(last_fname_numbers)
        last_fname_numbers = np.vstack((last_fname_numbers, -1.0*np.ones_like(last_fname_numbers))).T

        first_fname_numbers = np.concatenate(first_fname_numbers)
        first_fname_numbers = np.vstack((first_fname_numbers, 1.0*np.ones_like(first_fname_numbers))).T
        all_fname_numbers = np.vstack((first_fname_numbers, last_fname_numbers))
        all_fname_numbers = np.asarray(all_fname_numbers, dtype=np.float32)

        # Form training batches
        dataset = tf.data.Dataset.from_tensor_slices(all_fname_numbers)
        dataset = dataset.repeat(None)
        dataset = dataset.map(self.test_dataset_map,
                              num_parallel_calls=self.num_threads)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=3*batch_size)
        iterator = dataset.make_initializable_iterator()
        img1s, img2s, seg1s, fnames = iterator.get_next()
        if with_fname:
            return (img1s, img2s, seg1s, fnames), iterator
        return (img1s, img2s, seg1s), iterator


    def test_dataset_map(self, input_queue):
        fname_number, direction = input_queue[0], input_queue[1]
        t_shift = self.test_t_len
        img2_fname_number = t_shift * direction + fname_number
        # Conversions
        fname_number = tf.cast(fname_number, dtype=tf.int32)
        img2_fname_number = tf.cast(img2_fname_number, dtype=tf.int32)
        # Reading
        fname_1 = tf.gather(self.filenames, fname_number)
        fname_2 = tf.gather(self.filenames, img2_fname_number)
        annotation_fname = tf.gather(self.annotation_filenames, fname_number)

        file_content = tf.read_file(fname_1)
        image_1 = tf.image.decode_jpeg(file_content, channels=3)
        image_1 = self.preprocess_image(image_1)
        file_content = tf.read_file(fname_2)
        image_2 = tf.image.decode_jpeg(file_content, channels=3)
        image_2 = self.preprocess_image(image_2)
        file_content = tf.read_file(annotation_fname)
        seg_1 = tf.image.decode_jpeg(file_content, channels=1)
        seg_1 = self.preprocess_mask(seg_1)

        # Cropping preprocess
        image_1 = self.central_cropping(image_1, self.test_crop)
        image_2 = self.central_cropping(image_2, self.test_crop)
        seg_1 = self.central_cropping(seg_1, self.test_crop)

        return image_1, image_2, seg_1, fname_1

    def augmented_inputs(self, t_len=2, test_crops=[1.0]):
        # Generates multiple crop inputs for post processing.
        (img_1, img_2, seg_1, fname), itr = self.test_inputs(batch_size=1,
                                                             t_len=t_len,
                                                             with_fname=True,
                                                             test_crop=1.0)
        img_1 = tf.squeeze(img_1, axis=0)
        img_2 = tf.squeeze(img_2, axis=0)
        seg_1 = tf.squeeze(seg_1, axis=0)
        batch_dict = {'img_1s': {}, 'img_2s': {}, 'seg_1s': {}}
        for crop in test_crops:
            cropped_img_1 = self.central_cropping(img_1, cropping_percent=crop)
            cropped_img_2 = self.central_cropping(img_2, cropping_percent=crop)
            cropped_seg_1 = self.central_cropping(seg_1, cropping_percent=crop)
            batch_dict['seg_1s'][crop] = cropped_seg_1
            batch_dict['img_1s'][crop] = cropped_img_1
            batch_dict['img_2s'][crop] = cropped_img_2
        return batch_dict, fname, itr
