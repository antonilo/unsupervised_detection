import cv2
import gflags
import numpy as np
import os
import sys
import scipy.io as sio
import tensorflow as tf
from keras.utils.generic_utils import Progbar
from models.adversarial_learner import AdversarialLearner
from models.utils.general_utils import postprocess_mask, postprocess_image

from common_flags import FLAGS

des_width = 640
des_height = 384


def compute_IoU(gt_mask, pred_mask_f, threshold=0.1):
    gt_mask = gt_mask > (0.001)
    pred_mask = pred_mask_f > threshold
    pred_mask_compl = np.logical_not(pred_mask)


    union=float(np.sum(np.logical_or(gt_mask, pred_mask)))
    if union == 0.0:
        iou = 0.0
    else:
        iou = np.sum(np.logical_and(gt_mask, pred_mask)) / union

    union_compl=float(np.sum(np.logical_or(gt_mask, pred_mask_compl)))
    if union_compl == 0.0:
        iou_compl = 0.0
    else:
        iou_compl = np.sum(np.logical_and(gt_mask, pred_mask_compl)) / union_compl

    if iou > iou_compl:
        return iou, pred_mask_f
    else:
        return iou_compl, 1.0 - pred_mask_f

def compute_mae(gt_mask, pred_mask_f, threshold=0.1):
    pred_mask = np.asarray(pred_mask_f > threshold, dtype=np.float32)
    pred_mask_compl = np.asarray(np.logical_not(pred_mask), dtype=np.float32)

    mae = np.mean(np.abs(gt_mask - pred_mask))
    mae_compl = np.mean(np.abs(gt_mask - pred_mask_compl))
    mae = np.minimum(mae, mae_compl)
    return mae



def _test_masks():
    learner = AdversarialLearner()
    learner.setup_inference(FLAGS, aug_test=False)
    saver = tf.train.Saver([var for var in tf.trainable_variables()])
    CategoryIou = {}
    CategoryMae = {}
    # manages multi-threading
    sv = tf.train.Supervisor(logdir=FLAGS.test_save_dir,
                             save_summaries_secs=0,
                             saver=None)
    with sv.managed_session() as sess:
        checkpoint = FLAGS.ckpt_file
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("Resume model from checkpoint {}".format(checkpoint))
        else:
            raise IOError("Checkpoint file not found")

        sess.run(learner.test_iterator.initializer)

        n_steps = int(np.ceil(learner.test_samples / float(FLAGS.batch_size)))

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
            for batch_num in range(inference['input_image'].shape[0]):

                # select mask
                generated_mask = inference['gen_masks'][batch_num]
                gt_mask = inference['gt_masks'][batch_num]
                category = inference['img_fname'][batch_num].decode("utf-8").split('/')[-2]

                iou, out_mask = compute_IoU(gt_mask=gt_mask, pred_mask_f=generated_mask)
                mae = compute_mae(gt_mask=gt_mask, pred_mask_f=generated_mask)
                try:
                    CategoryIou[category].append(iou)
                    CategoryMae[category].append(mae)
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

                    preprocessed_bgr = postprocess_image(inference['input_image'][batch_num])
                    preprocessed_mask = postprocess_mask(out_mask)#(inference['gt_masks'][batch_num]>0.01)*1.0)

                    # Stich images
                    #results = np.hstack((
                    #    preprocessed_bgr,
                    #    preprocessed_mask,
                    #    # recall that now the gt_flow is an image
                    #    utils.postprocess_image(inference['gt_flow'][batch_num])))

                    # Overlap images
                    results = cv2.addWeighted(preprocessed_bgr, 0.5,
                                                  preprocessed_mask, 0.4, 0)
                    results = cv2.resize(results, (des_width, des_height))

                    cv2.imwrite(filename, results)

                    matlab_fname = os.path.join(save_dir,
                                            'result_{}.mat'.format(len(CategoryIou[category])))
                    sio.savemat(matlab_fname,
                                {'flow':inference['gt_flow'][batch_num],
                                 'img1':cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2RGB),
                                 'pred_mask': out_mask, #inference['gen_masks'][batch_num],
                                 'gt_mask': inference['gt_masks'][batch_num]} )
                i+=1

            progbar.update(step)

        tot_ious = 0
        tot_maes = 0
        per_cat_iou = []
        for cat, list_iou in CategoryIou.items():
            print("Category {}: IoU is {} and MAE is {}".format(cat, np.mean(list_iou), np.mean(CategoryMae[cat])))
            tot_ious += np.sum(list_iou)
            tot_maes += np.sum(CategoryMae[cat])
            per_cat_iou.append(np.mean(list_iou))
        print("The Average over the dataset: IoU is {} and MAE is {}".format(tot_ious/float(i), tot_maes/float(i)))
        print("The Average over sequences IoU is {}".format(np.mean(per_cat_iou)))
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
