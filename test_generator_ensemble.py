import cv2
import gflags
import numpy as np
import os
import sys
import scipy.io as sio
import tensorflow as tf
from keras.utils.generic_utils import Progbar
from test_generator import compute_mae, compute_IoU
from models.utils.general_utils import postprocess_mask, postprocess_image

from models.adversarial_learner import AdversarialLearner

from common_flags import FLAGS

des_width = 640
des_height = 384


def _test_masks():
    learner = AdversarialLearner()
    learner.setup_inference(FLAGS, aug_test=True)
    saver = tf.train.Saver([var for var in tf.trainable_variables()])
    CategoryIou = {}
    CategoryMae = {}
    # manages multi-threading
    sv = tf.train.Supervisor(logdir=FLAGS.test_save_dir,
                             save_summaries_secs=0,
                             saver=None)
    test_crops = learner.test_crops
    with sv.managed_session() as sess:
        checkpoint = FLAGS.ckpt_file
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("Resume model from checkpoint {}".format(checkpoint))
        else:
            raise IOError("Checkpoint file not found")

        sess.run(learner.test_iterator.initializer)

        n_steps = int(learner.test_samples)

        progbar = Progbar(target=n_steps)

        i = 0

        for step in range(n_steps):
            if sv.should_stop():
                break
            try:
                inference = learner.inference(sess)
            except tf.errors.OutOfRangeError:
                  print("End of testing dataset")  # ==> "End of dataset"
                  break
            # Now write images in the test folder
            outputs = inference['outs']
            fname = inference['img_fname']
            cropped_iou = []
            cropped_mae = []
            for crop in test_crops:
                # select mask
                generated_mask = outputs['pred_masks'][crop]
                gt_mask = outputs['gt_masks'][crop]

                iou, out_mask = compute_IoU(gt_mask=gt_mask, pred_mask_f=generated_mask)
                # Take the best one
                outputs['pred_masks'][crop] = out_mask
                mae = compute_mae(gt_mask=gt_mask, pred_mask_f=out_mask)
                cropped_iou.append(iou)
                cropped_mae.append(mae)

            cropped_iou = np.mean(cropped_iou)
            cropped_mae = np.mean(cropped_mae)

            category = fname.decode("utf-8").split('/')[-2]
            try:
                CategoryIou[category].append(cropped_iou)
                CategoryMae[category].append(cropped_mae)
            except:
                CategoryIou[category] = [iou]
                CategoryMae[category] = [mae]

            if FLAGS.generate_visualization:
            # Verbose image generation
                save_dir = os.path.join(FLAGS.test_save_dir, category)
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                filename = os.path.join(save_dir,
                                        "frame_{:08d}.png".format(len(CategoryIou[category])))

                # Take last one (for debugging only)
                preprocessed_bgr = postprocess_image(outputs['img_1s'][test_crops[-1]])
                preprocessed_mask = postprocess_mask(outputs['pred_masks'][test_crops[-1]])

                # Overlap images
                results = cv2.addWeighted(preprocessed_bgr, 0.5,
                                              preprocessed_mask, 0.4, 0)
                results = cv2.resize(results, (des_width, des_height))

                cv2.imwrite(filename, results)

                # Now write everything in the matlab file for post-processing
                matlab_fname = os.path.join(save_dir,
                                            'result_{}.mat'.format(len(CategoryIou[category])))
                matlab_out = {}
                for crop in test_crops:
                    matlab_out['img_1_{:03d}'.format(int(crop*100))] = outputs['img_1s'][crop]
                    matlab_out['pred_mask_{:03d}'.format(int(crop*100))] = outputs['pred_masks'][crop]
                    matlab_out['gt_mask_{:03d}'.format(int(crop*100))] = outputs['gt_masks'][crop]

                sio.savemat(matlab_fname, matlab_out)

            progbar.update(step)
            i+=1

        tot_ious = 0
        tot_maes = 0
        for cat, list_iou in CategoryIou.items():
            print("Category {}: IoU is {} and MAE is {}".format(cat, np.mean(list_iou), np.mean(CategoryMae[cat])))
            tot_ious += np.sum(list_iou)
            tot_maes += np.sum(CategoryMae[cat])
        print("The Average over the dataset: IoU is {} and MAE is {}".format(tot_ious/float(i), tot_maes/float(i)))
        print("Success: Processed {} frames".format(i))

def main(argv):
    # Utility main to load flags
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
      print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
      sys.exit(1)
    _test_masks()

if __name__ == "__main__":
    main(sys.argv)
