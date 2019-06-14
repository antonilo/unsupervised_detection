import numpy as np
import pydensecrf.densecrf as dcrf
import scipy.io as sio
import scipy
from scipy.misc import imresize
import os
from scipy.ndimage.filters import gaussian_filter

def run_crf(path_soft, sxy, srgb, scomp, gauss_k, out_path='./post_processed_davis'):

    result_path = path_soft
    seq_names = os.listdir(result_path)

    smooth_mask = gauss_k#1.0
    
    XY_deviation = sxy           # 10
    RGB_deviation = srgb         # 5
    Compatibility = scomp        # 1000
    Refine_num = 50              # 20

    Sum_iou = 0.0
    total_num = 0.0
    for seq in seq_names:
        seq_path = os.path.join( result_path, seq )
        seq_len = len( [name for name in os.listdir(seq_path) if name.endswith(".mat")] )
        out_dir = os.path.join( out_path, seq )
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        print(out_dir)
        for k in range(seq_len):
            r_name = os.path.join( seq_path, 'result_%(k)d.mat'%{"k":k+1} )

            total_num += 1.0
            result = sio.loadmat(r_name)

            pred_mask = np.float32(np.squeeze(result['pred_mask']))
            pred_f = np.float32(np.squeeze(result['running_avg_f']))
            pred_b = np.float32(np.squeeze(result['running_avg_b']))

            image = result['img1']
            gt_mask = result['gt_mask']
            gt_mask = np.float32(np.squeeze(gt_mask))

            objscore_m = np.sum(np.multiply(pred_mask,gt_mask)) / (np.sum(pred_mask)+1e-8)
            objscore_f = np.sum(np.multiply(pred_f   ,gt_mask)) / (np.sum(pred_f)+1e-8)
            objscore_b = np.sum(np.multiply(pred_b   ,gt_mask)) / (np.sum(pred_b)+1e-8)
            if objscore_m>=objscore_f and objscore_m>=objscore_b:
                mask = pred_mask
            elif objscore_f>=objscore_m and objscore_f>=objscore_b:
                mask = pred_f
            else:
                mask = pred_b

            mask_new, iou_new = refine(mask, image, gauss_k, sxy, srgb, scomp, gt_mask)

            new_name = os.path.join( out_dir, 'result_%(k)d.mat'%{"k":k+1} )
            sio.savemat( new_name, {'gt_mask':gt_mask, 'soft_mask':mask, 'mask':mask_new})

            Sum_iou += iou_new

    avg_iou = Sum_iou / total_num

    return avg_iou

def run_crf_original_resolution(path_soft, path_img, path_gt, sxy, srgb, scomp, gauss_k, out_path='./post_processed_davis_original'):
    result_path = path_soft
    seq_names = os.listdir(result_path)
    Sum_iou = 0.0
    total_num = 0.0
    for seq in seq_names:
        seq_path = os.path.join( result_path, seq )
        seq_len = len( [name for name in os.listdir(seq_path) if name.endswith(".mat")] )
        out_dir = os.path.join( out_path, seq )
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        print(out_dir)
        img_path = os.path.join(path_img, seq)
        gt_path  = os.path.join(path_gt,  seq)
        for k in range(seq_len):
            r_name = os.path.join( seq_path, 'result_%(k)d.mat'%{"k":k+1} )
            total_num += 1.0
            result = sio.loadmat(r_name)

            soft_mask = np.float32(np.squeeze(result['soft_mask']))

            imname = os.path.join( img_path, '%05d.jpg' % k )
            image = scipy.misc.imread(imname)
            gtname = os.path.join( gt_path, '%05d.png' % k )
            gt_mask = scipy.misc.imread(gtname) / 255.
            H, W = gt_mask.shape
            h, w = H*0.9, W*0.9
            h, w = int(h), int(w)
            soft_mask = imresize( soft_mask, (h,w) )
            soft_mask = soft_mask / (np.amax(soft_mask)+1e-8)
            mask = np.zeros((H,W))
            dh, dw = (H-h)/2, (W-w)/2
            mask[dh:dh+h, dw:dw+w] = soft_mask

            mask_new, iou_new = refine(mask, image, gauss_k, sxy, srgb, scomp, gt_mask)

            new_name = os.path.join( out_dir, 'result_%(k)d.mat'%{"k":k+1} )
            sio.savemat( new_name, {'mask':mask_new})

            Sum_iou += iou_new

    avg_iou = Sum_iou / total_num

    return avg_iou

def refine(mask, image, gk, sxy, srgb, compat, gtmask):
    dfield = dcrf.DenseCRF2D( mask.shape[1], mask.shape[0], 2 )

    U = gaussian_filter(mask, sigma=gk)
    U = U / (np.amax(U)+1e-8)
    U = np.clip(U, 1e-6, 1.0-1e-6)
    UU = np.zeros((2,mask.shape[0],mask.shape[1]))
    UU[1,:,:] = U
    UU[0,:,:] = 1.0-U        
    UU = -np.log(UU)
    UU = np.float32(UU)
    UU = UU.reshape((2,-1))
    dfield.setUnaryEnergy(UU)

    im = np.ascontiguousarray( image )
    dfield.addPairwiseBilateral(sxy=sxy, srgb=srgb, rgbim=im, compat=compat)
    Refine_num = 50
    Q = dfield.inference(Refine_num)
    new_mask = np.argmax(Q, axis=0).reshape((mask.shape[0],mask.shape[1]))
    new_mask = np.float32(new_mask)

    gt = gtmask > 0.1
    bmask = new_mask > 0.1

    inter_new = gt & bmask
    union_new = gt | bmask
    iou_new = np.float32(np.sum(inter_new)) / np.float32(np.sum(union_new))

    return new_mask, iou_new
