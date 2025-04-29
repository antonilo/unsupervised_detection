import os
import subprocess
import logging
import download_util


def main():
    # --- Fixed Parameters ---
    dataset_name = "FBMS"
    dataset_download_urls = [
        "https://lmb.informatik.uni-freiburg.de/resources/datasets/fbms/FBMS_Trainingset.zip",
        "https://lmb.informatik.uni-freiburg.de/resources/datasets/fbms/FBMS_Testset.zip",
    ]
    model_ckpt_basedir = (
        "fbms_best_model"  # this is the name under "unsupervised_detection_models"
    )
    rootdir_name = dataset_name

    LOG_LEVEL = logging.INFO
    TEST_CROP = 0.9  # FBMS default
    TEST_TEMPORAL_SHIFT = 1
    GENERATE_VISUALIZATION = True
    # --- End Fixed Parameters ---

    logging.basicConfig(level=LOG_LEVEL, format="[%(levelname)s] %(message)s")

    # --- Define Paths and URLs for FBMS-59 ---
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.abspath(
        os.path.join(script_dir, "..")
    )  # Go up one level from scripts/
    download_dir = os.path.join(base_dir, "download")
    results_dir = os.path.join(base_dir, "results", dataset_name)

    # Model Checkpoint
    model_ckpt_zip_url = "https://rpg.ifi.uzh.ch/data/unsupervised_detection_models.zip"
    model_ckpt_base = os.path.join(
        download_dir,
        "unsupervised_detection_models",
        model_ckpt_basedir,
        "model.best",
    )  # this will be used as the argument of test_generator.py
    model_ckpt_path = (
        model_ckpt_base + ".data-00000-of-00001"
    )  # actual checkpoint path.

    # PWCNet Checkpoint
    pwc_gdown_folder_url = (
        "https://drive.google.com/drive/folders/1gtGx_6MjUQC5lZpl6-Ia718Y_0pvcYou"
    )
    pwc_ckpt_path = os.path.join(
        download_dir,
        "pwcnet-lg-6-2-multisteps-chairsthingsmix",
        "pwcnet.ckpt-595000.data-00000-of-00001",
    )

    # --- Ensure Prerequisites ---
    logging.info(f"--- Checking Prerequisites for {dataset_name} ---")
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)  # Ensure results dir exists

    # 1. Dataset
    if not download_util.ensure_dataset(
        dataset_name,
        dataset_download_urls,
        download_dir,
    ):
        logging.error(f"Failed to prepare dataset {dataset_name}. Exiting.")
        exit(1)

    # 2. Model Checkpoint
    if not download_util.ensure_model_checkpoint(model_ckpt_zip_url, model_ckpt_path):
        logging.error(f"Failed to prepare model checkpoint {model_ckpt_path}. Exiting.")
        exit(1)

    # 3. PWCNet Checkpoint
    if not download_util.ensure_pwc_checkpoint(pwc_gdown_folder_url, pwc_ckpt_path):
        logging.error(f"Failed to prepare PWCNet checkpoint {pwc_ckpt_path}. Exiting.")
        exit(1)

    logging.info("--- Prerequisites Met ---")

    # --- Run Test Generator ---
    logging.info("Starting test generation...")
    test_command = [
        "python3",
        os.path.join(base_dir, "test_generator.py"),  # Path to test_generator.py
        f"--dataset={dataset_name}",
        f"--ckpt_file={model_ckpt_base}",
        f"--flow_ckpt={pwc_ckpt_path}",
        f"--test_crop={TEST_CROP}",
        f"--test_temporal_shift={TEST_TEMPORAL_SHIFT}",
        f"--root_dir={download_dir}/{rootdir_name}",
        f"--test_save_dir={results_dir}",
    ]
    if GENERATE_VISUALIZATION:
        test_command.append("--generate_visualization")

    logging.info(f"Running command: {' '.join(test_command)}")
    try:
        # Run the command from the script's directory
        subprocess.run(test_command, check=True, cwd=script_dir)
        logging.info("Test generation finished successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Test generation failed with error code {e.returncode}.")
        exit(1)
    except FileNotFoundError:
        logging.error(
            f"Error: test_generator.py not found in {script_dir}. Make sure it exists."
        )
        exit(1)


if __name__ == "__main__":
    main()
