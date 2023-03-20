import cv2
import gflags
import numpy as np
import os
import sys
import scipy.io as sio
import tensorflow as tf
from keras.utils.generic_utils import Progbar
from models.adversarial_learner import AdversarialLearner
from models.utils.general_utils import postprocess_mask, postprocess_image, compute_boundary_score

from common_flags import FLAGS

des_width = 640
des_height = 384
mask_threshold = 0.6

def _test_video():
    learner = AdversarialLearner()
    learner.setup_inference(FLAGS, aug_test=False)
    saver = tf.train.Saver([var for var in tf.trainable_variables()])
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
                category = inference['img_fname'][batch_num].decode("utf-8").split('/')[-2]

                if FLAGS.generate_visualization:
                    # Verbose image generation
                    save_dir = os.path.join(FLAGS.test_save_dir, category)
                    if not os.path.isdir(save_dir):
                        os.mkdir(save_dir)
                    filename = os.path.join(save_dir,
                                              "frame_{:08d}.png".format(i)

                    preprocessed_bgr = postprocess_image(inference['input_image'][batch_num])
                    preprocessed_mask = postprocess_mask(generated_mask)

                    # Overlap images
                    results = cv2.addWeighted(preprocessed_bgr, 0.5,
                                                  preprocessed_mask, 0.4, 0)
                    results = cv2.resize(results, (des_width, des_height))

                    cv2.imwrite(filename, results)

                i+=1

            progbar.update(step)

        print("Success: Processed {} frames".format(i))

def main(argv):
    # Utility main to load flags
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
      print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
      sys.exit(1)
    _test_video()

if __name__ == "__main__":
    main(sys.argv)
