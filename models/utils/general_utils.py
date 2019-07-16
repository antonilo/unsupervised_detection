import numpy as np
import cv2
import tensorflow as tf

def postprocess_flow(flow):
    """
    Function to visualize the flow.
    Args:
        flow : [H,W,2] optical flow
    Returs:
        grayscale image to visualize flow
    """
    flow = flow[:,:,0] # do it dirty, ony first channel
    min_flow = np.min(flow)
    rescaled = flow + min_flow
    max_rescaled = np.max(rescaled)
    normalized = rescaled / max_rescaled
    normalized = np.asarray(normalized / max_rescaled * 255, np.uint8)
    normalized = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)

    return normalized

def postprocess_image(image):
    """
    Function to un-normalize images.
    Args:
        flow : [H,W,3] image
    Returs:
        grayscale image to visualize flow
    """
    p_image = image + 0.5
    un_normalized = np.asarray(p_image * 255, np.uint8)
    un_normalized = cv2.cvtColor(un_normalized, cv2.COLOR_RGB2BGR)

    return un_normalized

def postprocess_mask(mask):
    """
    Function to un-normalize images.
    Args:
        flow : [H,W,3] image
    Returs:
        grayscale image to visualize flow
    """
    # We want it in red
    un_normalized = np.asarray(mask * 255.0, np.uint8)
    tile = np.zeros_like(un_normalized, dtype=np.uint8)
    un_normalized = np.concatenate((tile, un_normalized, tile), axis=-1)
    #un_normalized = cv2.cvtColor(un_normalized, cv2.COLOR_RGB2BGR)

    return un_normalized

def generate_error_map(image, losses, box_lenght):
    """
    Function to overlap an error map to an image
    Args:
        image: input image
        losses: list of losses, one for each masked part of the flow.
    Returs:
        error_map: overlapped error_heatmap and image.
    """
    box_lenght = int(box_lenght)

    # Assert that everything is correct
    num_boxes = int(image.shape[0] / box_lenght) * int(image.shape[1] / box_lenght)
    assert(num_boxes ==len(losses))

    img_width = int(np.floor(image.shape[1] / box_lenght) * box_lenght)
    img_height = int(np.floor(image.shape[0] / box_lenght) * box_lenght)
    image = image[:img_height, :img_width]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    heatmap = np.ones_like(image[:,:,0])
    res_heatmap = np.reshape(heatmap, (box_lenght, box_lenght, num_boxes))
    res_heatmap = res_heatmap * np.array(losses)
    heatmap = np.zeros((img_height, img_width))
    # ugly for loop, unable to solve atm
    i = 0
    for y in np.arange(0, img_height, step=box_lenght):
        for x in np.arange(0, img_width, step=box_lenght):
            # convert to x,y coordinates
            heatmap[y: y+box_lenght, x: x+box_lenght] = res_heatmap[:,:,i]
            i+=1
    heatmap = np.asarray(heatmap / np.max(heatmap) * 255, dtype=np.uint8)
    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    final = cv2.addWeighted(heatmap_img, 0.5, postprocess_image(image), 0.5, 0)

    return final

def tf_iou_computation(gt_masks, pred_masks):

    epsilon = tf.constant(1e-8) # To avoid division by zero
    pred_masks = tf.cast(pred_masks, tf.bool)
    union=tf.reduce_sum(tf.cast(tf.logical_or(gt_masks, pred_masks),
                                dtype=tf.float32), axis=[1,2,3]) + epsilon
    IoU = tf.reduce_sum(tf.cast(tf.logical_and(gt_masks, pred_masks),
                                dtype=tf.float32), axis=[1,2,3]) / union

    return IoU

def disambiguate_forw_back(pred_masks, threshold=0.1):
    border_th = tf.constant(0.6)
    # Might be redundant but makes no assumption 
    pred_masks = tf.cast(pred_masks > threshold, tf.float32)
    pred_masks_compl = 1.0 - pred_masks
    scores = compute_boundary_score_tf(pred_masks)
    scores = tf.reshape(scores, [-1,1,1,1]) < border_th
    scores = tf.cast(scores, tf.float32)
    forward_masks = scores * pred_masks + (1.0 - scores) * pred_masks_compl
    return forward_masks

def compute_all_IoU(pred_masks, gt_masks, threshold=0.1):
    gt_masks= gt_masks > 0.01
    object_masks = disambiguate_forw_back(pred_masks, threshold)
    IoU = tf_iou_computation(gt_masks=gt_masks, pred_masks=object_masks)
    return IoU

def compute_boundary_score(segmentation):
    """
    This score indicates how many image borders the segmentation
    mask occupies. If lower than a threshold, then it indicates foreground,
    else background. The threshold is generally set to 0.6, which means
    that to be background, the mask has to (approx.) occupy more than two borders.
    """
    H = segmentation.shape[0]
    W = segmentation.shape[1]
    up_bord = segmentation[0:2, :]
    bottom_bord = segmentation[H-2:H, :]
    left_bord = segmentation[:, 0:2]
    right_bord = segmentation[:, W-2:W]
    border_occ = np.sum(up_bord)+np.sum(bottom_bord)+np.sum(left_bord)+np.sum(right_bord)
    border_occ /= 1.0*(up_bord.size+bottom_bord.size+left_bord.size+right_bord.size)
    return border_occ

def compute_boundary_score_tf(segmentation):
    """
    Same as above but in tensorflow"
    """
    height, width = segmentation.get_shape().as_list()[1:3]
    up_bord = segmentation[:,0:2, :, :]
    bottom_bord = segmentation[:,height-2:height,:,:]
    width_bord_size = 2.0 * width
    left_bord = segmentation[:, :, 0:2, :]
    right_bord = segmentation[:, :, width-2:width, :]
    height_bord_size = 2.0 * height
    border_occ = tf.reduce_sum(up_bord, axis=[1,2,3]) + \
                 tf.reduce_sum(bottom_bord, axis=[1,2,3]) + \
                 tf.reduce_sum(left_bord, axis=[1,2,3]) + \
                 tf.reduce_sum(right_bord, axis=[1,2,3]) # [B]
    border_occ /= (2*width_bord_size + 2*height_bord_size)
    return border_occ
