import os
import sys
import scipy.io as sio
import numpy as np
import pyflow
import cv2
from scipy.misc import imresize

seq_names = [ 'soapbox', 'scooter-black', 'parkour', 'paragliding-launch',
              'motocross-jump', 'libby', 'kite-surf', 'horsejump-high', 'goat',
              'drift-straight', 'drift-chicane', 'dog', 'dance-twirl', 'cows',
              'car-shadow', 'car-roundabout', 'camel', 'breakdance', 'bmx-trees',
              'blackswan' ]
seq_num = [99, 43, 100, 80, 40, 49, 50, 50, 90, 50, 52, 60, 90, 104, 40, 75, 90, 84, 80, 50]

def buffer_to_soft_score( buffer_path,
                          out_path,
                          max_shift=2,
                          base_crop=90.0,
                          seq_names=seq_names,
                          seq_num=seq_num,
                          dprefix='davis_shift' ):
    result_path = buffer_path

    start_crop = 85
    end_crop = 100
    base_H = 192
    base_W = 384
    san_t = 0.6

    for i in range(len(seq_names)):
        out_dir = os.path.join(out_path, seq_names[i])
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        print(out_dir)
        for k in range(seq_num[i]):
            k += 1

            score = []
            img1 = []
            gt_mask = []
            for shift in range(max_shift):
                shift += 1

                r_name_b = os.path.join( result_path, '%s_%d' % (dprefix, -shift), seq_names[i], 'result_%d.mat' % (k) )
                r_name_f = os.path.join( result_path, '%s_%d' % (dprefix,  shift), seq_names[i], 'result_%d.mat' % (k) )
                r_b = sio.loadmat(r_name_b)
                r_f = sio.loadmat(r_name_f)
                for crop in range(start_crop, end_crop+1, 5):

                    s_name = 'pred_mask_%03d' % (crop)
                    s_b = np.squeeze( r_b[s_name] )
                    s_f = np.squeeze( r_f[s_name] )

                    sani_b = sanity_check(s_b)
                    sani_f = sanity_check(s_f)
                    if sani_b >= san_t and sani_f >= san_t:
                        s_b = s_b * 0.0
                        s_f = s_f * 0.0
                    elif sani_b >= san_t and sani_f < san_t:
                        s_b = s_f
                    elif sani_b < san_t and sani_f >= san_t:
                        s_f = s_b
                    else:
                        pass

                    if shift==1 and crop==base_crop:
                        if score == []:
                            score = s_b + s_f
                        else:
                            score += s_b + s_f
                        im1_name = 'img_1_%03d' % (crop)
                        img1 = r_f[im1_name]
                        img1 = ( (img1+0.5)*255 ).astype('uint8')
                        gt_name = 'gt_mask_%03d' % (crop)
                        gt_mask = r_f[gt_name]
                    else:
                        ratio = crop / base_crop
                        rec_s_b = rectify_pred_mask( s_b, ratio, base_H, base_W )
                        rec_s_f = rectify_pred_mask( s_f, ratio, base_H, base_W )
                        if score == []:
                            score = rec_s_b + rec_s_f
                        else:
                            score += rec_s_b + rec_s_f

            min_score = np.amin(score)
            max_score = np.amax(score)
            pred_mask = (score-min_score) / (max_score-min_score+1e-6)

            out_name = os.path.join( out_dir, 'result_%d.mat' % (k) )

            sio.savemat( out_name, {'pred_mask':pred_mask, 'img1':img1, 'gt_mask':gt_mask} )

    propagate(out_path, seq_names, seq_num)

def rectify_pred_mask( pred_mask, crop, H, W ):
    ## rectify predicitons made on different crops
    if crop > 1:
        crop = 1.0/crop
        hh = int(H*crop)
        ww = int(W*crop)
        h = int( (H-hh)/2 )
        w = int( (W-ww)/2 )
        pred_crop = pred_mask[ h:h+hh, w:w+ww ]
        rec_pred_mask = imresize( pred_crop, (H,W) )
    else:
        rec_pred_mask = np.zeros((H,W))
        hh = int(H*crop)
        ww = int(W*crop)
        pred_crop = imresize( pred_mask, (hh,ww) )
        h = max( int((H-hh)/2), 0 )
        w = max( int((W-ww)/2), 0 )
        rec_pred_mask[ h:h+hh, w:w+ww ] = pred_crop
    return rec_pred_mask / ( np.amax(rec_pred_mask) + 1e-6 )

def sanity_check(s):
    H = s.shape[0]
    W = s.shape[1]
    a = s[0:2, :]
    b = s[H-2:H, :]
    c = s[:, 0:2]
    d = s[:, W-2:W]
    sanity = np.sum(a)+np.sum(b)+np.sum(c)+np.sum(d)
    sanity /= 1.0*(a.size+b.size+c.size+d.size)
    return sanity

def propagate(out_path, seq_names, seq_num):
    # Computes a moving average of masks based on optical flow.

    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    w_r = 0.85
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    ## forward pass
    for i in range(len(seq_names)):
        out_dir = os.path.join(out_path, seq_names[i])
        print(out_dir)
        running_avg = []
        for k in range(seq_num[i]):
            k += 1

            if k==1:
                r_name = os.path.join( out_dir, 'result_%d.mat' % (k) )
                r = sio.loadmat(r_name)
                running_avg = np.squeeze(r['pred_mask'])
                r['running_avg_f'] = running_avg
                sio.savemat( r_name, r )
                continue

            r2_name = os.path.join( out_dir, 'result_%d.mat' % (k) )
            r1_name = os.path.join( out_dir, 'result_%d.mat' % (k-1) )
            r2 = sio.loadmat(r2_name)
            r1 = sio.loadmat(r1_name)

            I2 = (np.squeeze( r2['img1'] )).astype(float) / 255.
            I1 = (np.squeeze( r1['img1'] )).astype(float) / 255.
            I2 = I2.copy(order='C')
            I1 = I1.copy(order='C')
            u, v, im2W = pyflow.coarse2fine_flow(
                             I2, I1, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
                             nSORIterations, colType)

            flow = np.concatenate((u[..., None], v[..., None]), axis=2)
            h, w = flow.shape[:2]
            flow[:,:,0] += np.arange(w)
            flow[:,:,1] += np.arange(h)[:,np.newaxis]

            flow = flow.astype(np.float32)
            s1 = np.squeeze( r1['pred_mask'] )
            s2 = cv2.remap( s1, flow, None, cv2.INTER_LINEAR)
            s2 = ( s2 / (np.amax(s2)+1e-8) )
            running_avg = cv2.remap( running_avg, flow, None, cv2.INTER_LINEAR)
            running_avg = ( running_avg / (np.amax(running_avg)+1e-8) )
            running_avg = (1-w_r)*s2 + w_r*running_avg
            running_avg = ( running_avg / (np.amax(running_avg)+1e-8) )

            r2['running_avg_f'] = running_avg
            sio.savemat( r2_name, r2 )

    ## backward pass
    for i in range(len(seq_names)):
        out_dir = os.path.join(out_path, seq_names[i])
        print(out_dir)
        running_avg = []
        for k in range(seq_num[i]):
            k = seq_num[i] - k

            if k==seq_num[i]:
                r_name = os.path.join( out_dir, 'result_%d.mat' % (k) )
                r = sio.loadmat(r_name)
                running_avg = np.squeeze(r['pred_mask'])
                r['running_avg_b'] = running_avg
                sio.savemat( r_name, r )
                continue

            r1_name = os.path.join( out_dir, 'result_%d.mat' % (k) )
            r2_name = os.path.join( out_dir, 'result_%d.mat' % (k+1) )
            r1 = sio.loadmat(r1_name)
            r2 = sio.loadmat(r2_name)

            I2 = (np.squeeze( r2['img1'] )).astype(float) / 255.
            I1 = (np.squeeze( r1['img1'] )).astype(float) / 255.
            I2 = I2.copy(order='C')
            I1 = I1.copy(order='C')
            u, v, im2W = pyflow.coarse2fine_flow(
                             I1, I2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
                             nSORIterations, colType)

            flow = np.concatenate((u[..., None], v[..., None]), axis=2)
            h, w = flow.shape[:2]
            flow[:,:,0] += np.arange(w)
            flow[:,:,1] += np.arange(h)[:,np.newaxis]

            flow = flow.astype(np.float32)
            s2 = np.squeeze( r2['pred_mask'] )
            s1 = cv2.remap( s2, flow, None, cv2.INTER_LINEAR)
            s1 = ( s1 / (np.amax(s1)+1e-8) )
            running_avg = cv2.remap( running_avg, flow, None, cv2.INTER_LINEAR)
            running_avg = ( running_avg / (np.amax(running_avg)+1e-8) )
            running_avg = (1-w_r)*s1 + w_r*running_avg
            running_avg = ( running_avg / (np.amax(running_avg)+1e-8) )

            r1['running_avg_b'] = running_avg
            sio.savemat( r1_name, r1 )

