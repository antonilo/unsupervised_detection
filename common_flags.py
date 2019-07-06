import gflags
FLAGS = gflags.FLAGS


# Train parameters
gflags.DEFINE_integer('img_width', 384, 'Target Image Width')
gflags.DEFINE_integer('img_height', 192, 'Target Image Height')
gflags.DEFINE_integer('batch_size', 16, 'Batch size in training and evaluation')
gflags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
gflags.DEFINE_float("flow_normalizer", 80.0, "Scale for regularization of flow")
gflags.DEFINE_integer("max_epochs", 40, "Maximum number of training epochs")
gflags.DEFINE_integer("num_samples_train", 5000, "number of samples per epoch, "
                      "not necessarly corresponding to the number of training samples.")
gflags.DEFINE_float("train_crop", 0.9, "minimum random cropping percentage of input images")
gflags.DEFINE_integer("max_temporal_len", 2, "Maximum delta time for image 2")
gflags.DEFINE_integer("min_temporal_len", 1, "Minimum delta time for image 2")
gflags.DEFINE_float("cbn", 0.5, "power to square loss (0.5 for L1, 1. for L2)")
gflags.DEFINE_float("epsilon", 75.0, "epsilon in reduction rate computation")
gflags.DEFINE_integer("iters_rec", 1, "training iteration of recover per step."
                      " Increase this if not using a pre-trained checkpoint")
gflags.DEFINE_integer("iters_gen", 3, "training iteration of generator per step")
gflags.DEFINE_integer('num_threads', 6, 'Number of threads reading and '
                      '(optionally) preprocessing input files into queues')
gflags.DEFINE_bool('resume_train', False, 'Whether to restore a trained'
                   ' model for training')

# Path Parameters
gflags.DEFINE_string('root_dir',"/your/path/to/DAVIS_2016", 'Folder containig the evaluation dataset')
gflags.DEFINE_string('train_partition', 'trainval', 'Training Partition to be used')
gflags.DEFINE_string('dataset', 'DAVIS2016', 'Dataset used for evaluation. '
                     ' Either SEGTRACK or FBMS or DAVIS2016')
gflags.DEFINE_string('recover_ckpt', "", 'Checkpoint of the pre-trained recover.'
                     ' If None, it will train the recover from scratch.')
gflags.DEFINE_string('flow_ckpt', "", 'Checkpoint to the pre-trained PWCNet')
gflags.DEFINE_string('full_model_ckpt', "", 'File containing'
                     ' the checkpoint of the entire network. '
                     'Use this flag if you want to resume a training.')
gflags.DEFINE_string('checkpoint_dir', "", "Experiment folder. It will contain"
                     "the saved checkpoints and tensorboard logs.")

# Log parameters
gflags.DEFINE_integer("summary_freq", 30,
                      "Logging tensorboard summaries every summary_freq iterations")
gflags.DEFINE_integer("save_freq", 5,
                      "Save the latest model every save_freq epochs")

# Testing parameters
gflags.DEFINE_bool('generate_visualization', False, "Whether to save images while computing metrics")
gflags.DEFINE_float("test_crop", 0.9, "central cropping percentages of input images at test time")
gflags.DEFINE_integer('test_temporal_shift', 1,
                      'Constant Temporal shift between the two images used to calculate flow images.')
gflags.DEFINE_string("ckpt_file", "", "Model Checkpoint to be used for testing.")
gflags.DEFINE_string("test_partition", "val", "Can be train/val/trainval")
gflags.DEFINE_string('test_save_dir', "",
                     "Test Folder for the experiment. It can store generated predictions and logs")
