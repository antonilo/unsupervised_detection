"""
This File implements the data reader for the FBMS59
Dataset. See the file davis2016_data_utils.py for
a more detailed documentation of the functions.
The main difference with respect to DAVIS2016 is
the fact that the data reader returns the number
of images per category (used to explain away the
large class imbalance of this dataset in score computation).
After the first use, you can speed up the code by commenting
pre-processing away (See line 109).
"""
import numpy as np
import os
import cv2
import re
import tensorflow as tf
from data.aug_flips import random_flip_images

class DirectoryIterator(object):
    """
    Class for managing data loading.of images and labels
    We assume that the folder structure is:

    """
    def __init__(self, directory, part='train', for_testing=False, test_temporal_t=1):
        self.directory = directory
        self.num_experiments = 0
        self.samples_per_cat = {}

        parsing_dir ={'train': ['Trainingset'],
                      'val' : ['Testset'],
                      'trainval': ['Trainingset', 'Testset']}

        data_dirs = [os.path.join(directory, d) for d in parsing_dir.get(part)]
        for d in data_dirs:
            if not os.path.isdir(d):
                raise IOError("Directory {} file not found".format(d))

        # First count how many experiments are out there
        self.samples = 0
        # Each filename is a tuple image / components
        self.image_filenames = []
        self.annotation_filenames = []
        for d in data_dirs:
            if for_testing:
                self._parse_testtime_dir(d, test_temporal_t)
            else:
                self._parse_data_dir(d)

        if self.samples == 0:
            raise IOError("Did not find any file in the dataset folder")
        if not for_testing:
            self.num_experiments = len(self.image_filenames)

        print('Found {} images belonging to {} experiments.'.format(
                self.samples, self.num_experiments))


    def _parse_data_dir(self, data_dir):
        """
        This function will read all the files in data_dir and return a list of
        lists containing the different fnames for each category.
        """
        categories = os.listdir(data_dir)
        for folder_name in categories:
            all_fnames_list_fname = os.path.join(data_dir, folder_name,
                                                 folder_name + ".bmf")
            if not os.path.isfile(all_fnames_list_fname):
                raise IOError("Not found file {}".format(all_fnames_list_fname))
            all_fnames_list = np.loadtxt(all_fnames_list_fname, dtype=np.str,
                                         skiprows=1)
            # Correct from pgm to jpg
            all_fnames_list = [f.split('.')[0]+'.jpg' for f in all_fnames_list]

            all_fnames_list = [os.path.join(data_dir, folder_name, f) for f \
                               in all_fnames_list]

            self.samples += len(all_fnames_list)
            # Append the last
            self.image_filenames.append(all_fnames_list)

    def _parse_testtime_dir(self, data_dir, test_temporal_t=1):
        """
        This function will read all the files in data_dir and return a list of
        lists containing the different fnames for each category.
        """
        self.test_tuples = []
        categories = os.listdir(data_dir)
        for folder_name in categories:
            all_fnames_list_fname = os.path.join(data_dir, folder_name,
                                                 folder_name + ".bmf")
            if not os.path.isfile(all_fnames_list_fname):
                raise IOError("Not found file {}".format(all_fnames_list_fname))
            all_fnames_list = np.loadtxt(all_fnames_list_fname, dtype=np.str,
                                         skiprows=1)
            # Correct from pgm to jpg
            all_fnames_list = [f.split('.')[0]+'.jpg' for f in all_fnames_list]

            all_fnames_list = [os.path.join(data_dir, folder_name, f) for f \
                               in all_fnames_list]
            # Get ground_truth
            annotation_fnames, numbers, type_weird = self.find_gt(os.path.join(data_dir,
                                                                   folder_name,
                                                                   'GroundTruth'))
            goal_annotation_fnames = [f.split('.')[0] + '.jpg' for f in annotation_fnames]
            goal_annotation_fnames = [os.path.join(data_dir, folder_name, 'GroundTruth', f) for f \
                                 in goal_annotation_fnames]

            # NOTE: Run the commented part only once to preprocess GT
            annotation_fnames = [os.path.join(data_dir, folder_name, 'GroundTruth', f) for f \
                                 in annotation_fnames]
            for i in range(len(goal_annotation_fnames)):
                mask = cv2.imread(annotation_fnames[i])
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask = mask / 255.0
                if type_weird:
                    mask[mask>0.99] = 0.0
                if 'marple7' == folder_name:
                    mask = mask>0.05
                elif 'marple2' == folder_name:
                    mask = mask>0.4
                else:
                    mask = mask>0.1
                mask = np.asarray(mask*255, dtype=np.uint8)
                cv2.imwrite(goal_annotation_fnames[i], mask)

            # Create offsets
            numbers = np.array(numbers) - np.min(numbers)
            seq_len = np.max(numbers)
            offsets = numbers + test_temporal_t
            if offsets[0] < numbers[0]:
                # test was negative, needs to increase:
                offsets[0] += 2*abs(test_temporal_t)
            if offsets[-1] > numbers[-1]:
                # test was positive, needs to decrease:
                offsets[-1] -= 2*abs(test_temporal_t)

            for i in range(len(offsets)):
                offsets[i] = np.maximum(offsets[i], 0)
                offsets[i] = np.minimum(offsets[i], seq_len)


            for i, k in enumerate(numbers):
                self.test_tuples.append((all_fnames_list[k], all_fnames_list[offsets[i]],
                                         goal_annotation_fnames[i], "{}".format(len(annotation_fnames))))

            self.samples += len(annotation_fnames)
            self.samples_per_cat[folder_name] = len(annotation_fnames)
            self.num_experiments+=1


    def find_gt(self, directory):
        all_files = os.listdir(directory)
        # Check in which kind of folder you are
        type_weird=False
        for file in all_files:
            if file.endswith('ppm'):
                type_weird=True
                break
        if not type_weird:
            all_files = [file for file in all_files if file.endswith('pgm')]
            # Sort them
            try:
                all_files = sorted(all_files, key=lambda x: int(x.split('.')[0].split('_')[-1]))
                numbers = [int(file.split('.')[0].split('_')[-1]) for file in all_files]
            except:
                all_files = sorted(all_files, key=lambda x: int(re.search(r'\d+', x).group()))
                numbers = [int(re.search(r'\d+', file).group()) for file in all_files]
            return all_files, numbers, type_weird
        # Solve weird type
        all_files = [file for file in all_files if file.endswith('ppm') and not 'PROB' in file]
        all_files = sorted(all_files, key=lambda x: int(x.split('_')[1]))
        numbers = [int(file.split('_')[1]) for file in all_files]
        return all_files, numbers, type_weird

class FBMS59Reader(object):
    def __init__(self, root_dir, max_temporal_len=3, min_temporal_len=2,
                 num_threads=6):
        self.root_dir = root_dir
        self.max_temporal_len = max_temporal_len
        self.min_temporal_len = min_temporal_len
        assert min_temporal_len < max_temporal_len, "Temporal lenghts are not consistent"
        assert min_temporal_len > 0, "Min temporal len should be positive"
        self.num_threads = num_threads

    def get_filenames_list(self, partition):
        iterator = DirectoryIterator(self.root_dir, partition)
        filenames, annotation_filenames = iterator.image_filenames, \
                                            iterator.annotation_filenames
        #Training calls it before, so it will be overwritten
        self.val_samples = iterator.samples
        return filenames, annotation_filenames

    def get_test_tuples(self, partition, test_temporal_t=1):
        iterator = DirectoryIterator(self.root_dir, partition, for_testing=True,
                                     test_temporal_t=test_temporal_t)
        test_tuples = iterator.test_tuples
        #Training calls it before, so it will be overwritten
        self.val_samples = iterator.samples
        self.samples_per_cat = iterator.samples_per_cat
        self.num_categories = len(iterator.samples_per_cat.keys())
        return test_tuples

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

    def image_inputs(self, batch_size=32, partition='train',
                     train_crop=1.0):
        # Generates input batches for FBMS dataset.
        t_len = self.max_temporal_len
        file_list, _ = self.get_filenames_list(partition)
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
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=3*batch_size)
        iterator = dataset.make_initializable_iterator()
        img1s, img2s = iterator.get_next()
        # Extra arguments returned for compatibility with test functions.
        return (img1s, img2s, tf.constant(1.0),'f', 1.0), iterator

    def test_inputs(self, batch_size=32, partition='val',
                    t_len=2, with_fname=False, test_crop=1.0):
        # Reads test inputs data
        # The main difference with Davis2016 consists in retuning
        # the number of elements per category.
        test_tuples = self.get_test_tuples(partition, t_len)
        self.test_crop = test_crop
        self.num_threads = 1
        # Form training batches
        dataset = tf.data.Dataset.from_tensor_slices(test_tuples)
        dataset = dataset.repeat(None)
        dataset = dataset.map(self.test_dataset_map,
                              num_parallel_calls=1)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(buffer_size=3*batch_size)
        iterator = dataset.make_initializable_iterator()
        img1s, img2s, seg1s, fnames, samples_per_cat = iterator.get_next()
        if with_fname:
            return (img1s, img2s, seg1s, fnames, samples_per_cat), iterator
        return (img1s, img2s, seg1s, samples_per_cat), iterator

    def test_dataset_map(self, input_queue):
        fname_1, fname_2, annotation_fname, samples_per_cat = input_queue[0],\
            input_queue[1], input_queue[2], input_queue[3]
        samples_per_cat = tf.string_to_number(samples_per_cat)
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

        return image_1, image_2, seg_1, fname_1, samples_per_cat

    def augmented_inputs(self, partition='val', t_len=2,
                         test_crops=[1.0]):
        (img_1, img_2, seg_1, fname, _), itr = self.test_inputs(batch_size=1,
                                                             t_len=t_len,
                                                             partition=partition,
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
