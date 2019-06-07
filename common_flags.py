import gflags
FLAGS = gflags.FLAGS


# Train parameters
gflags.DEFINE_integer('img_width', 384, 'Target Image Width')
gflags.DEFINE_integer('img_height', 192, 'Target Image Height')
gflags.DEFINE_integer('batch_size', 16, 'Batch size in training and evaluation')
gflags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
gflags.DEFINE_float("flow_normalizer", 100.0, "Scale for regularization of flow")   ### davis:80 ###
gflags.DEFINE_integer("max_epochs", 40, "Maximum number of training epochs")
gflags.DEFINE_integer("num_samples_train", 5000, "number of training samples")
# Cropping parameter
gflags.DEFINE_float("train_crop", 0.9, "minimum random cropping percentage of input images")
gflags.DEFINE_float("test_crop", 0.9, "central cropping percentage of input images at test time")

gflags.DEFINE_integer("max_temporal_len", 2, "Maximum delta time for image 2")   ### davis:2 ###
gflags.DEFINE_integer("min_temporal_len", 1, "Maximum delta time for image 2")   ### davis:2 ###
gflags.DEFINE_float("cbn", 0.5, "power to square loss (0.5 for L1, 1. for L2)")
gflags.DEFINE_float("epsilon", 75.0, "epsilon in reduction rate computation")
gflags.DEFINE_integer("iters_rec", 1, "training iteration of recover per step")  ### davis:1 ###
gflags.DEFINE_integer("iters_gen", 3, "training iteration of generator per step") ### davis:3 ###

gflags.DEFINE_integer('num_threads', 6, 'Number of threads reading and '
                      '(optionally) preprocessing input files into queues')

gflags.DEFINE_string('root_dir',   "/home/ycyang/Datasets/SegTrackv2", 'File containing'
                     ' tf recorded validation experiments')
gflags.DEFINE_string('train_partition', 'trainval', 'Training Partition to be used')
gflags.DEFINE_string('dataset', 'SEGTRACK', 'Either SEGTRACK or FBMS or DAVIS2016')
gflags.DEFINE_string('recover_ckpt', "./tests/model_tests/aug_davis_pretrain_normalized_flow/model-175", 'File containing'
                     ' the checkpoint of the recover (inpaint) network')
gflags.DEFINE_string('flow_ckpt', "./models/PWCNet/checkpoint/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000", 'File containing'
                     ' the checkpoint of the network')
gflags.DEFINE_string('full_model_ckpt', "", 'File containing'
                     ' the checkpoint of the entire network to be restored')

gflags.DEFINE_string('checkpoint_dir', "./tests/model_tests/davis2017_test", "Directory name to"
                     "save checkpoints and logs.")

# Log parameters
gflags.DEFINE_bool('resume_train', False, 'Whether to restore a trained'
                   ' model for training')
gflags.DEFINE_integer("summary_freq", 30, "Logging every log_freq iterations")
gflags.DEFINE_integer("save_freq", 5,
                      "Save the latest model every save_freq epochs")

# Testing parameters
gflags.DEFINE_bool('generate_visualization', False, "Whether to save images while computing metrics")
gflags.DEFINE_string('test_save_dir', "./tests/fbms_save",
                     'TF record containing testing samples')
gflags.DEFINE_string("ckpt_file", "/home/ycyang/Documents/ADMoseg/best_segtrack_models/2018-11-12_0.51_100fn_n2t/model.best", #"./tests/model_tests/FBMS59/fbms_FN100_E75_T2/model-16",
                     "Checkpoint file")
gflags.DEFINE_integer('test_temporal_shift', -2, ' Constant Temporal shift used to calculate flow images (constant)')
gflags.DEFINE_string("test_partition", "val",
                     "Can be train/val/trainval")
