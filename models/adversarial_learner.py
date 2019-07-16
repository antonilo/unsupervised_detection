import os
import sys
import time
from itertools import count
import math
import numpy as np
import tensorflow as tf
from keras.utils.generic_utils import Progbar
from .nets import recover_net, generator_net
from .utils.loss_utils import charbonnier_loss, train_op
from .utils.flow_utils import flow_to_image_tf, preprocess_flow_batch
from .utils.general_utils import compute_all_IoU, disambiguate_forw_back
from .PWCNet.model_pwcnet import ModelPWCNet
from data.davis2016_data_utils import Davis2016Reader
from data.fbms_data_utils import FBMS59Reader
from data.segtrackv2_data_utils import SegTrackV2Reader

class AdversarialLearner(object):
    def __init__(self):
        pass

    def load_training_data(self):
        with tf.name_scope("data_loading"):
            if self.config.dataset== 'DAVIS2016':
                reader = Davis2016Reader(self.config.root_dir,
                                     max_temporal_len=self.config.max_temporal_len,
                                     min_temporal_len=self.config.min_temporal_len,
                                     num_threads=self.config.num_threads)

                train_batch, train_iter = reader.image_inputs(batch_size=self.config.batch_size,
                                                              train_crop=self.config.train_crop,
                                                              partition=self.config.train_partition)

                val_batch, val_iter = reader.test_inputs(batch_size=self.config.batch_size,
                                                         t_len=self.config.test_temporal_shift,
                                                         test_crop=self.config.test_crop,
                                                         partition='val')
            elif self.config.dataset == 'FBMS':
                reader = FBMS59Reader(self.config.root_dir,
                                      max_temporal_len=self.config.max_temporal_len,
                                      min_temporal_len=self.config.min_temporal_len,
                                      num_threads=self.config.num_threads)
                train_batch, train_iter = reader.image_inputs(batch_size=self.config.batch_size,
                                                              train_crop=self.config.train_crop,
                                                              partition=self.config.train_partition)

                val_batch, val_iter = reader.test_inputs(batch_size=self.config.batch_size,
                                                         t_len=self.config.test_temporal_shift,
                                                         test_crop=self.config.test_crop,
                                                         with_fname=True,
                                                         partition='val')
                self.num_categories = reader.num_categories

            elif self.config.dataset == 'SEGTRACK':
                reader = SegTrackV2Reader(self.config.root_dir,
                                          max_temporal_len=self.config.max_temporal_len,
                                          min_temporal_len=self.config.min_temporal_len,
                                          num_threads=self.config.num_threads)
                train_batch, train_iter = reader.image_inputs(batch_size=self.config.batch_size,
                                                              train_crop=self.config.train_crop)

                val_batch, val_iter = reader.test_inputs(batch_size=self.config.batch_size,
                                                         t_len=self.config.test_temporal_shift,
                                                         test_crop=self.config.test_crop)

            else:
                raise IOError("Dataset should be DAVIS2016 / FBMS / SEGTRACK")

        self.num_samples_val = reader.val_samples
        return train_batch, val_batch, train_iter, val_iter

    def build_train_graph(self):
        is_training_ph = tf.placeholder(tf.bool, shape=(), name="is_training")

        train_batch, val_batch, train_iter, val_iter = self.load_training_data()

        current_batch = tf.cond(is_training_ph, lambda: train_batch,
                                lambda: val_batch)

        image_batch, images_2_batch = current_batch[0], current_batch[1]

        # Flow Computation (On original image sizes, 640x398)
        flow_network = ModelPWCNet()
        flow_batch = flow_network.predict_from_img_pairs(image_batch, images_2_batch)

        # Reshape everything to desired image size
        image_batch = tf.image.resize_images(image_batch, [self.config.img_height,
                                                           self.config.img_width])
        flow_batch = tf.image.resize_images(flow_batch, [self.config.img_height,
                                                           self.config.img_width])
        # Reshape mask to correct ratio (will be used only for evaluation)
        gt_masks = tf.image.resize_images(val_batch[2], [self.config.img_height,
                                                         self.config.img_width],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # Normalize flow by a constant
        flow_batch = flow_batch / tf.constant(self.config.flow_normalizer)

        with tf.name_scope("MaskNet") as scope:
            # This is the generator network
            generated_masks = generator_net(images=image_batch,
                                       flows=preprocess_flow_batch(flow_batch),
                                       training = is_training_ph,
                                       scope=scope,
                                       reuse=False)

            complementary_masks = (1.0 - generated_masks)

        flow_masked = flow_batch * (1.0 - generated_masks)
        flow_complementary_mask = flow_batch * (1.0 - complementary_masks)

        with tf.name_scope("FlownetS") as scope:
            # This is the inpainter network
            pred_flows = recover_net(image_batch,
                                     flow_masked,
                                     mask=generated_masks,
                                     scope=scope,
                                     reuse=False)

            pred_complementary_flows = recover_net(image_batch,
                                                   flow_complementary_mask,
                                                   mask=complementary_masks,
                                                   scope=scope,
                                                   reuse=True)

            # This is used for normalization, flow from single image.
            pred_flow_from_image = recover_net(image_batch,
                                               tf.zeros_like(flow_batch), # No flow is passed
                                               mask=tf.ones_like(generated_masks), # Entire image masked
                                               scope=scope,
                                               reuse=True)

        # Compute IoU of the generated masks ( for validation only, will not be
        # executed during training)
        all_IoU = compute_all_IoU(pred_masks=generated_masks,
                                  gt_masks=gt_masks)
        sum_IoU = tf.reduce_sum(all_IoU)

        # Define now all training losses.

        losses = {}

        # Compute the loss for the recover and its complementary.
        rec_loss = charbonnier_loss(pred_flows=pred_flows,
                                    gt_flows=flow_batch,
                                    masks=generated_masks,
                                    cbn=self.config.cbn) # [B,]
        # Complementary
        rec_compl_loss = charbonnier_loss(pred_flows=pred_complementary_flows,
                                          gt_flows=flow_batch,
                                          masks=complementary_masks,
                                          cbn=self.config.cbn) #[B,]

        # Recover loss also includes the flow from single image error.
        # See paper for details.

        recover_red_rate = tf.reduce_sum(rec_loss)
        recover_red_rate_compl = tf.reduce_sum(rec_compl_loss)

        # Mask on entire image means keeping means summing over every pixel
        image_prior_decoder= charbonnier_loss(gt_flows=flow_batch,
                                              pred_flows=pred_flow_from_image,
                                              masks=tf.ones_like(flow_batch),
                                              cbn=self.config.cbn)
        image_prior_decoder = tf.reduce_sum(image_prior_decoder)

        num_pixels =  tf.constant(self.config.img_width * \
                                  self.config.img_height * self.config.batch_size,
                                  dtype=tf.float32)

        recover_loss = (recover_red_rate + recover_red_rate_compl + \
                        image_prior_decoder) / num_pixels

        # Compute the loss for the generator and its complementary.
        # Epsilon is a tuned parameter which avoids division by zero
        # and pushes the generator away from mask all/nothing local minima.

        epsilon = tf.constant(self.config.epsilon)
        den_red = charbonnier_loss(gt_flows=flow_batch,
                                   pred_flows=pred_flow_from_image,
                                   masks=generated_masks,
                                   cbn=self.config.cbn) + epsilon
        red_rate_object = 1.0 - rec_loss / den_red
        red_rate_object = tf.reduce_mean(red_rate_object, axis=[0])
        # Complementary
        den_red_compl = charbonnier_loss(gt_flows=flow_batch,
                                         pred_flows=pred_flow_from_image,
                                         masks=complementary_masks,
                                         cbn=self.config.cbn) + epsilon
        red_rate_compl = 1.0 - rec_compl_loss / den_red_compl
        red_rate_compl = tf.reduce_mean(red_rate_compl, axis=[0])

        # Generator loss is the quality of flow reconstruction.
        generator_loss = red_rate_object + red_rate_compl

        losses['generator'] = generator_loss
        losses['recover'] = recover_loss
        losses['red_rate'] = red_rate_object
        losses['red_rate_compl'] = red_rate_compl
        # First element of the batch, for visualization
        losses['reconstruction_loss'] = rec_loss[0]
        losses['reconstruction_compl_loss'] = rec_compl_loss[0]
        losses['denominator_red_rate'] = den_red[0]
        losses['denominator_red_rate_compl'] = den_red_compl[0]

        with tf.name_scope("train_op"):
            self.global_step = tf.Variable(0,
                                           name='global_step',
                                           trainable=False)

            recover_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             'FlownetS')
            generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               'MaskNet')

            optimizer  = tf.train.AdamOptimizer(learning_rate=1e-4,
                                             beta1=self.config.beta1, epsilon=1e-8)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            # The can_flag puts some noise into the gradients if the latter are vanishing.
            # This usually happen when the generator encounters the local minimum of
            # masking everything or masking nothing.
            self.train_generator_op, self.generator_var_grads = train_op(loss=losses['generator'],
                                               var_list=generator_vars,
                                               optimizer=optimizer,
                                               gradient_clip_value=.2,
                                               can_change=True)

            self.train_recover_op, self.recover_var_grads = train_op(loss=losses['recover'],
                                             var_list=recover_vars,
                                             optimizer=optimizer,
                                             gradient_clip_value=0.2,
                                             can_change=False)

            self.train_generator_op = tf.group([self.train_generator_op, update_ops])
            self.train_recover_op = tf.group([self.train_recover_op, update_ops])

            self.incr_global_step = tf.assign(self.global_step,
                                              self.global_step+1)
        self.iterators = [train_iter, val_iter]
        self.image_batch = image_batch
        self.image_2_batch = images_2_batch
        self.losses = losses
        # Names are used for post-processing
        self.val_cat = val_batch[-1]
        self.val_names = val_batch[-2]
        self.generated_masks = generated_masks
        self.flow_masked = flow_masked
        self.flow_gt_batch = flow_batch
        self.pred_flow = pred_flows*generated_masks + flow_batch * (1-generated_masks)
        self.pred_flow_compl = pred_flows*complementary_masks + flow_batch * (1-complementary_masks)
        self.is_training = is_training_ph
        self.train_steps_per_epoch = \
            int(math.ceil(self.config.num_samples_train/self.config.batch_size))
        self.val_steps_per_epoch = int(np.ceil(float(self.num_samples_val) / self.config.batch_size))
        self.val_iou = sum_IoU
        self.all_iou = all_IoU

    def collect_summaries(self):
        """Collects all summaries to be shown in the tensorboard"""
        for key, value in self.losses.items():
            tf.summary.scalar(key, value, collections=['step_sum'])

        tf.summary.image("input_image", self.image_batch, max_outputs=1,
                         collections=['step_sum'])
        tf.summary.image("next_image", self.image_2_batch, max_outputs=1,
                         collections=['step_sum'])
        tf.summary.image("masked_flow",
                         (flow_to_image_tf(self.flow_gt_batch))* \
                         (1.0 - disambiguate_forw_back(self.generated_masks)),
                         max_outputs=1, collections=['step_sum'])
        tf.summary.image("PWC_Flow",
                         flow_to_image_tf(self.flow_gt_batch),
                         max_outputs=1, collections=['step_sum'])
        tf.summary.image("Rec_flow",
                         flow_to_image_tf(self.pred_flow),
                         max_outputs=1, collections=['step_sum'])
        tf.summary.image("Rec_flow_compl",
                         flow_to_image_tf(self.pred_flow_compl),
                         max_outputs=1, collections=['step_sum'])

        # Log Gradients
        for grad, var in self.recover_var_grads:
            tf.summary.histogram(var.op.name + "/gradients", grad,
                                 collections=["step_sum"])
        for grad, var in self.generator_var_grads:
            tf.summary.histogram(var.op.name + "/gradients", grad,
                                 collections=["step_sum"])

        self.step_sum = tf.summary.merge(tf.get_collection('step_sum'))
        #######################
        # VALIDATION ERROR SUM#
        #######################
        self.val_iou_ph = tf.placeholder(tf.float32, [])
        tf.summary.scalar("IoU on Validation", self.val_iou_ph,
                          collections = ["validation_summary"])
        self.val_sum = tf.summary.merge(tf.get_collection('validation_summary'))

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to {}/model-{}".format(checkpoint_dir,
                                                             step))
        if step == 'best':
            self.saver.save(sess,
                        os.path.join(checkpoint_dir, model_name + '.best'))
        else:
            self.saver.save(sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def train(self, config):
        """High level train function.
        Args:
            config: Configuration dictionary
        Returns:
            None
        """
        self.config = config
        self.build_train_graph()
        self.collect_summaries()
        self.min_val_iou = -1.0e12 # Initialize to large value
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                        for v in tf.trainable_variables()])
        self.saver = tf.train.Saver([var for var in \
            tf.trainable_variables()] + [self.global_step], max_to_keep=40)
        # Load pre-trained recover net
        recover_saver = tf.train.Saver(tf.trainable_variables(scope='FlownetS'))
        # Load flow predictor
        flow_saver = tf.train.Saver(tf.global_variables(scope='pwcnet'))

        sv = tf.train.Supervisor(logdir=config.checkpoint_dir,
                                 save_summaries_secs=0,
                                 saver=None)

        with sv.managed_session() as sess:
            print("Number of params: {}".format(sess.run(parameter_count)))
            if os.path.isfile(self.config.flow_ckpt + ".index"):
                flow_saver.restore(sess, self.config.flow_ckpt)
                print("Flow net loaded from {}".format(self.config.flow_ckpt))
            else:
                raise IOError("Could not find flow ckpt file. Aborting.")

            if config.resume_train:
                if os.path.isfile(self.config.full_model_ckpt + ".index"):
                    checkpoint = self.config.full_model_ckpt
                elif os.path.isdir(self.config.checkpoint_dir):
                    checkpoint = tf.train.latest_checkpoint(
                                                self.config.checkpoint_dir)
                assert checkpoint, "Found no checkpoint to resume training!"
                self.saver.restore(sess, checkpoint)
                print("Resumed training from model {}".format(checkpoint))
            elif os.path.isfile(self.config.recover_ckpt + ".index"):
                print("Recover net loaded from previous ckpt")
                recover_saver.restore(sess, self.config.recover_ckpt)
            else:
                # Better to initialize form a recover pretrained on simulated datasets.
                # This can be downloaded from the project page.
                print("No recover checkpoint found! Train Recover from Scratch")

            progbar = Progbar(target=self.train_steps_per_epoch)

            for it in self.iterators:
                sess.run(it.initializer)

            iters_rec = self.config.iters_rec
            iters_gen = self.config.iters_gen

            print("-------------------------------------")
            print("Training {} Recover and {} Generator".format(iters_rec, iters_gen))
            print("-------------------------------------")

            sum_iters = iters_rec + iters_gen

            for step in count(start=1):
                if sv.should_stop():
                    break
                start_time = time.time()
                fetches = {"global_step": self.global_step}

                if step % sum_iters == 0:
                    # Global step increased after every cycle
                    fetches.update({"incr_global_step": self.incr_global_step})

                if (step % sum_iters) < iters_rec:
                    fetches["train_op"] = self.train_recover_op
                else:
                    fetches["train_op"] = self.train_generator_op

                if (step % config.summary_freq ==0):
                    fetches["loss_recover"] = self.losses['recover']
                    fetches["loss_generator"] = self.losses['generator']
                    fetches["summary"] = self.step_sum

                results = sess.run(fetches,
                                   feed_dict={ self.is_training : True })

                progbar.update(step % self.train_steps_per_epoch)
                gs = results["global_step"]

                if step % config.summary_freq == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil( step /self.train_steps_per_epoch)
                    train_step = step - (train_epoch - 1) * self.train_steps_per_epoch
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it loss_generator: %4.4f loss_recover %4.4f" \
                       % (train_epoch, train_step, self.train_steps_per_epoch, \
                                time.time() - start_time, \
                          results["loss_generator"], results["loss_recover"]))

                if step % self.train_steps_per_epoch == 0:
                    # This differ from the last when resuming training
                    train_epoch = int(step / self.train_steps_per_epoch)
                    progbar = Progbar(target=self.train_steps_per_epoch)
                    self.epoch_end_callback(sess, sv, train_epoch)
                    if (train_epoch == self.config.max_epochs):
                        print("-------------------------------")
                        print("Training completed successfully")
                        print("-------------------------------")
                        break

    def epoch_end_callback(self, sess, sv, epoch_num):
        # Evaluate val loss
        validation_iou = 0
        print("\nComputing Validation IoU")
        progbar = Progbar(target=self.val_steps_per_epoch)

        for i in range(self.val_steps_per_epoch):
            loss_iou = sess.run(self.val_iou,
                             feed_dict={self.is_training: False})
            validation_iou+= loss_iou
            progbar.update(i)
        validation_iou /= self.val_steps_per_epoch*self.config.batch_size

        # Log to Tensorflow board
        val_sum = sess.run(self.val_sum, feed_dict ={
                           self.val_iou_ph: validation_iou})

        sv.summary_writer.add_summary(val_sum, epoch_num)

        print("Epoch [{}] Validation IoU: {}".format(
            epoch_num, validation_iou))
        # Model Saving
        if validation_iou > self.min_val_iou:
            self.save(sess, self.config.checkpoint_dir, 'best')
            self.min_val_iou = validation_iou
        if epoch_num % self.config.save_freq == 0:
            self.save(sess, self.config.checkpoint_dir, epoch_num)

    def build_test_graph(self):
        """This graph will be used for testing. In particular, it will
           compute the loss on a testing set, or some other utilities.
        """
        with tf.name_scope("data_loading"):
            if self.config.dataset== 'DAVIS2016':
                reader = Davis2016Reader(self.config.root_dir, num_threads=1)
                test_batch, test_iter = reader.test_inputs(batch_size=self.config.batch_size,
                                                      t_len=self.config.test_temporal_shift,
                                                      with_fname=True,
                                                      test_crop=self.config.test_crop,
                                                      partition=self.config.test_partition)

            elif self.config.dataset == 'FBMS':
                reader = FBMS59Reader(self.config.root_dir)
                test_batch, test_iter = reader.test_inputs(batch_size=self.config.batch_size,
                                                           test_crop=self.config.test_crop,
                                                           t_len=self.config.test_temporal_shift,
                                                           with_fname=True,
                                                           partition=self.config.test_partition)
            elif self.config.dataset == 'SEGTRACK':
                reader = SegTrackV2Reader(self.config.root_dir, num_threads=1)
                test_batch, test_iter = reader.test_inputs(batch_size=self.config.batch_size,
                                                           test_crop=self.config.test_crop,
                                                           t_len=self.config.test_temporal_shift,
                                                           with_fname=True)
            else:
                raise IOError("Dataset should be DAVIS2016 / FBMS / SEGTRACK")

            image_batch, images_2_batch, gt_mask_batch, fname_batch = test_batch[0], \
                                            test_batch[1], test_batch[2], test_batch[3]

        # Flow computed on original image size
        flow_network = ModelPWCNet()
        flow_batch = flow_network.predict_from_img_pairs(image_batch, images_2_batch)

        # Reshape everything
        image_batch = tf.image.resize_images(image_batch, [self.config.img_height,
                                                           self.config.img_width])
        flow_batch = tf.image.resize_images(flow_batch, [self.config.img_height,
                                                           self.config.img_width])
        # Reshape mask to correct ratio
        gt_mask_batch = tf.image.resize_images(gt_mask_batch, [self.config.img_height,
                                                         self.config.img_width],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # Normalize flow
        flow_batch = flow_batch / tf.constant(self.config.flow_normalizer)

        with tf.name_scope("MaskNet") as scope:
            generated_masks = generator_net(images=image_batch,
                                       flows=preprocess_flow_batch(flow_batch),
                                       training=False,
                                       scope=scope,
                                       reuse=False)

        flow_masked = flow_batch * (1.0 - generated_masks)

        with tf.name_scope("FlownetS") as scope:
            pred_flows = recover_net(image_batch,
                                     flow_masked,
                                     mask=generated_masks,
                                     scope=scope,
                                     reuse=False)


        self.input_image = image_batch
        self.gt_flow = flow_batch
        self.fname_batch = fname_batch
        self.generated_masks = generated_masks
        self.test_samples = reader.val_samples
        self.gt_masks = gt_mask_batch
        self.pred_flow = pred_flows
        self.test_iterator = test_iter

    def build_aug_test_graph(self):
        """This graph will be used for the generation of the results
           with multiple crop of the images.
           This improves the results while doing ensembling.
           Requires batch size to be one (automatically handled)
        """
        test_crops = [0.85, 0.9, 0.95, 1.0]
        print("Evaluating the following crops {}".format(test_crops))
        with tf.name_scope("data_loading"):
            if self.config.dataset == 'DAVIS2016':
                reader = Davis2016Reader(self.config.root_dir)
                test_batch, fname_batch, test_iter = reader.augmented_inputs(
                                                      t_len=self.config.test_temporal_shift,
                                                      test_crops=test_crops,
                                                      partition=self.config.test_partition)

            elif self.config.dataset == 'FBMS':
                assert 'FBMS' in self.config.root_dir
                reader = FBMS59Reader(self.config.root_dir)
                test_batch, fname_batch, test_iter = reader.augmented_inputs(t_len=self.config.test_temporal_shift,
                                                                test_crops=test_crops,
                                                                partition=self.config.test_partition)
            elif self.config.dataset == 'SEGTRACK':
                reader = SegTrackV2Reader(self.config.root_dir)
                test_batch, fname_batch, test_iter = reader.augmented_inputs(t_len=self.config.test_temporal_shift,
                                                                test_crops=test_crops)
            else:
                raise IOError("Dataset should be DAVIS2016 / FBMS / SEGTRACK")

        results = {'pred_masks': {}, 'gt_masks': {}, 'img_1s': {}}
        for crop in test_crops:
            image_batch = tf.expand_dims(test_batch['img_1s'][crop], axis=0)
            images_2_batch = tf.expand_dims(test_batch['img_2s'][crop], axis=0)
            gt_mask_batch = test_batch['seg_1s'][crop]

            flow_network = ModelPWCNet()
            flow_batch = flow_network.predict_from_img_pairs(image_batch, images_2_batch)

            # Reshape everything
            image_batch = tf.image.resize_images(image_batch, [self.config.img_height,
                                                               self.config.img_width])
            flow_batch = tf.image.resize_images(flow_batch, [self.config.img_height,
                                                               self.config.img_width])
            # Reshape mask to correct ratio
            gt_mask_batch = tf.image.resize_images(gt_mask_batch, [self.config.img_height,
                                                             self.config.img_width],
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            # Normalize flow
            flow_batch = flow_batch / tf.constant(self.config.flow_normalizer)

            generated_masks = generator_net(images=image_batch,
                                       flows=preprocess_flow_batch(flow_batch),
                                       training=False,
                                       scope="MaskNet/",
                                       reuse=tf.AUTO_REUSE)


            results['pred_masks'][crop] = tf.squeeze(generated_masks, axis=0)
            results['gt_masks'][crop] = gt_mask_batch
            results['img_1s'][crop] = tf.squeeze(image_batch, axis=0)


        self.outputs = results
        self.fname_batch = tf.squeeze(fname_batch)
        self.test_samples = reader.val_samples
        self.test_crops = test_crops
        self.test_iterator = test_iter

    def setup_inference(self, config, aug_test=False):
        """Sets up the inference graph.
        Args:
            config: config dictionary.
        """
        self.config = config
        self.aug_test = aug_test
        if self.aug_test:
            self.build_aug_test_graph()
        else:
            self.build_test_graph()

    def inference(self, sess):
        """Outputs a dictionary with the results of the required operations.
        Args:
            sess: current session
        Returns:
            results: dictionary with output of testing operations.
        """
        if self.aug_test:
            fetches = {'outs': self.outputs, 'img_fname': self.fname_batch}
        else:

            fetches = { 'gen_masks': self.generated_masks, 'pred_flow': self.pred_flow,
                        'input_image': self.input_image, 'gt_flow': self.gt_flow,
                       'gt_masks': self.gt_masks, 'img_fname': self.fname_batch}

        results = sess.run(fetches)

        return results
