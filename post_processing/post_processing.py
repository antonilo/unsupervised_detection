import generate_soft_score_from_buffer
import os
import crf_refine

# Path to the buffer generated with the file scripts/generate_buffer_DAVIS2016.sh
path_buffer = '/tmp/buffer_davis'
# Output folder for averaged masks over several crops and time differences
out_soft_score = './soft_davis'
# Whether to produce results on original image size for benchmarking with SOTA
benchmark=False

if not os.path.isdir(out_soft_score):
    os.mkdir(out_soft_score)
    print(out_soft_score)
generate_soft_score_from_buffer.buffer_to_soft_score( buffer_path=path_buffer, out_path=out_soft_score )


path_soft = out_soft_score
# Output folder for predictions after CRF on training image size (384 x 192)
resized_out = './crf_resized_davis'
if not os.path.isdir(resized_out):
    os.mkdir(resized_out)
    print(resized_out)
sxy = 25.
srgb = 5.
scomp = 5.
gauss_k = 0.1
iou_resized = crf_refine.run_crf(path_soft, sxy, srgb, scomp, gauss_k, out_path=resized_out)
print('iou of the resized version:')
print(iou_resized)

# Output folder for predictions after CRF on original image size (854x480) used
# for benchmarking.
# An extra post-processing step selects the best detection candidate from the
# set of predicted connected masks, by measuring overlap with the GT mask.
if benchmark:
    path_img='./DAVIS/JPEGImages/480p' # Dataset images path
    path_gt='./DAVIS/Annotations/480p' # GT annotations path
    original_out = './crf_original_davis'
    sxy = 60.
    if not os.path.isdir(original_out):
        os.mkdir(original_out)
        print(original_out)
    iou_original = crf_refine.run_crf_original_resolution(resized_out, path_img, path_gt, sxy, srgb, scomp, gauss_k, original_out)
    print('iou of the original resolution version:')
    print(iou_original)
