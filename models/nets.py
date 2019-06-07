import tensorflow as tf
from .utils.convolution_utils import gen_conv, gen_deconv, conv, deconv

def generator_net(images, flows, scope, reuse=None, training=True):
    """Mask network.
    Args:
        image: input rgb image [-0.5, +0.5]
        flows: rgb flow image masked [-0.5, +0.5]
    Returns:
        mask: mask region [0, 1], 1 is fully masked, 0 is not.
    """

    mask_channels = 2 # probability of 1 and zero
    x = tf.concat((images, flows), 3)  #[B, H, W, 5]

    cnum = 32
    with tf.variable_scope(scope, reuse=reuse):
        # stage1
        x_0 = gen_conv(x, cnum, 5, 1, name='conv1', training=training) # ---------------------------
        x   = gen_conv(x_0, 2*cnum, 3, 2, name='conv2_downsample', training=training) # Skip connection
        x_1 = gen_conv(x, 2*cnum, 3, 1, name='conv3', training=training) # -------------------
        x   = gen_conv(x_1, 4*cnum, 3, 2, name='conv4_downsample', training=training)
        x   = gen_conv(x, 4*cnum, 3, 1, name='conv5', training=training)
        x_2 = gen_conv(x, 4*cnum, 3, 1, name='conv6', training=training) # -----------------
        x   = gen_conv(x_2, 4*cnum, 3, rate=2, name='conv7_atrous', training=training)
        x   = gen_conv(x, 4*cnum, 3, rate=4, name='conv8_atrous', training=training)
        x   = gen_conv(x, 4*cnum, 3, rate=8, name='conv9_atrous', training=training)
        x   = gen_conv(x, 4*cnum, 3, rate=16, name='conv10_atrous', training=training)
        x   = gen_conv(x, 4*cnum, 3, 1, name='conv11', training=training) + x_2 #-------------
        x   = gen_conv(x, 4*cnum, 3, 1, name='conv12', training=training)
        x   = gen_deconv(x, 2*cnum, name='conv13_upsample', training=training)
        x   = gen_conv(x, 2*cnum, 3, 1, name='conv14', training=training) + x_1 # --------------------
        x   = gen_deconv(x, cnum, name='conv15_upsample', training=training) + x_0 #-------------------
        x   = gen_conv(x, cnum//2, 3, 1, name='conv16', training=training)
        x   = gen_conv(x, mask_channels, 3, 1, activation=tf.identity,
                     name='conv17', training=training)
        # Division by constant experimentally improved training
        x = tf.divide(x, tf.constant(10.0))
        generated_mask = tf.nn.softmax(x, axis=-1)
        # get logits for probability 1
        generated_mask = tf.expand_dims(generated_mask[:,:,:,0], axis=-1)
        return generated_mask


def recover_net( img1, flow_masked, mask, scope, reuse=None, f=0.25, training=True ):
    batch_size = tf.shape(img1)[0]
    C = flow_masked.get_shape().as_list()[-1]
    orisize = img1.get_shape().as_list()[1:-1]

    ones_x = tf.ones_like(flow_masked)[:, :, :, 0:1]
    # Augmentation of the flow
    flow_masked = tf.concat([flow_masked, ones_x, 1.0-mask], axis=3)
    flow_in_channels = flow_masked.get_shape().as_list()[-1]

    with tf.variable_scope(scope, reuse=reuse):

        aconv1 = conv( img1,    'aconv1', shape=[7,7, 3,  int(64*f)],  stride=2, reuse=reuse, training=training ) # h/2(192), 64
        aconv2 = conv( aconv1,  'aconv2', shape=[5,5,int(64*f), int(128*f)],  stride=2, reuse=reuse, training=training ) # h/4(96),  128
        aconv3 = conv( aconv2,  'aconv3', shape=[5,5,int(128*f),int(256*f)],  stride=2, reuse=reuse, training=training ) # h/8(48),  256
        aconv31= conv( aconv3, 'aconv31', shape=[3,3,int(256*f),int(256*f)],  stride=1, reuse=reuse, training=training )
        aconv4 = conv( aconv31, 'aconv4', shape=[3,3,int(256*f),int(512*f)],  stride=2, reuse=reuse, training=training ) # h/16(24), 512
        aconv41= conv( aconv4, 'aconv41', shape=[3,3,int(512*f),int(512*f)],  stride=1, reuse=reuse, training=training )
        aconv5 = conv( aconv41, 'aconv5', shape=[3,3,int(512*f),int(512*f)],  stride=2, reuse=reuse, training=training ) # h/32(12), 512
        aconv51= conv( aconv5, 'aconv51', shape=[3,3,int(512*f),int(512*f)],  stride=1, reuse=reuse, training=training )
        aconv6 = conv( aconv51, 'aconv6', shape=[3,3,int(512*f),int(512*f)],  stride=2, reuse=reuse, training=training ) # h/64(6),  512

        bconv1 = conv( flow_masked,    'bconv1', shape=[7,7, flow_in_channels,  int(64*f)],  stride=2, reuse=reuse, training=training ) # h/2(192), 64
        bconv2 = conv( bconv1,  'bconv2', shape=[5,5,int(64*f), int(128*f)],  stride=2, reuse=reuse, training=training ) # h/4(96),  128
        bconv3 = conv( bconv2,  'bconv3', shape=[5,5,int(128*f),int(256*f)],  stride=2, reuse=reuse, training=training ) # h/8(48),  256
        bconv31= conv( bconv3, 'bconv31', shape=[3,3,int(256*f),int(256*f)],  stride=1, reuse=reuse, training=training )
        bconv4 = conv( bconv31, 'bconv4', shape=[3,3,int(256*f),int(512*f)],  stride=2, reuse=reuse, training=training ) # h/16(24), 512
        bconv41= conv( bconv4, 'bconv41', shape=[3,3,int(512*f),int(512*f)],  stride=1, reuse=reuse, training=training )
        bconv5 = conv( bconv41, 'bconv5', shape=[3,3,int(512*f),int(512*f)],  stride=2, reuse=reuse, training=training ) # h/32(12), 512
        bconv51= conv( bconv5, 'bconv51', shape=[3,3,int(512*f),int(512*f)],  stride=1, reuse=reuse, training=training )
        bconv6 = conv( bconv51, 'bconv6', shape=[3,3,int(512*f),int(512*f)],  stride=2, reuse=reuse, training=training ) # h/64(6),  512

        #conv6 = tf.add( aconv6, bconv6 )
        conv6 = tf.concat( (aconv6, bconv6), 3 )  #h/64(6) 512*2*f
        outsz = bconv51.get_shape()                              # h/32(12), 512*f
        deconv5 = deconv( conv6, size=[outsz[1],outsz[2]], name='deconv5', shape=[4,4,int(512*2*f),int(512*f)], reuse=reuse, training=training )
        concat5 = tf.concat( (deconv5,bconv51,aconv51), 3 )              # h/32(12), 512*3*f

        flow5 = conv( concat5, 'flow5', shape=[3,3,int(512*3*f),C], stride=1, reuse=reuse, training=training, activation=tf.identity ) # h/32(12), C
        outsz = bconv41.get_shape()                              # h/16(24), 512*f
        deconv4 = deconv( concat5, size=[outsz[1],outsz[2]], name='deconv4', shape=[4,4,int(512*3*f),int(512*f)], reuse=reuse, training=training )
        upflow4 = deconv( flow5,   size=[outsz[1],outsz[2]], name='upflow4', shape=[4,4,C,C], reuse=reuse, training=training, activation=tf.identity )
        concat4 = tf.concat( (deconv4,bconv41,aconv41,upflow4), 3 )      # h/16(24), 512*3*f+C

        flow4 = conv( concat4, 'flow4', shape=[3,3,int(512*3*f+C),C], stride=1, reuse=reuse, training=training, activation=tf.identity ) # h/16(24), C
        outsz = bconv31.get_shape()                              # h/8(48),  256*f
        deconv3 = deconv( concat4, size=[outsz[1],outsz[2]], name='deconv3', shape=[4,4,int(512*3*f+C),int(256*f)], reuse=reuse, training=training )
        upflow3 = deconv( flow4,   size=[outsz[1],outsz[2]], name='upflow3', shape=[4,4,C,C], reuse=reuse, training=training, activation=tf.identity )
        concat3 = tf.concat( (deconv3,bconv31,aconv31,upflow3), 3 )      # h/8(48),  256*3*f+C

        flow3 = conv( concat3, 'flow3', shape=[3,3,int(256*3*f+C),C], stride=1, reuse=reuse, training=training, activation=tf.identity ) # h/8(48), C
        outsz = bconv2.get_shape()                               # h/4(96),  128*f
        deconv2 = deconv( concat3, size=[outsz[1],outsz[2]], name='deconv2', shape=[4,4,int(256*3*f+C),int(128*f)], reuse=reuse, training=training )
        upflow2 = deconv( flow3,   size=[outsz[1],outsz[2]], name='upflow2', shape=[4,4,C,C], reuse=reuse, training=training, activation=tf.identity )
        concat2 = tf.concat( (deconv2,bconv2,aconv2,upflow2), 3 )       # h/4(96),  128*3*f+C

        flow2 = conv( concat2, 'flow2', shape=[3,3,int(128*3*f+C),C], stride=1, reuse=reuse, training=training, activation=tf.identity ) # h/4(96), C
        outsz = bconv1.get_shape()                               # h/2(192), 64*f
        deconv1 = deconv( concat2, size=[outsz[1],outsz[2]], name='deconv1', shape=[4,4,int(128*3*f+C),int(64*f)], reuse=reuse, training=training )
        upflow1 = deconv( flow2,   size=[outsz[1],outsz[2]], name='upflow1', shape=[4,4,C,C], reuse=reuse, training=training, activation=tf.identity )
        concat1 = tf.concat( (deconv1,bconv1,aconv1,upflow1), 3 )       # h/2(192), 64*3*f+C

        flow1 = conv( concat1, 'flow1', shape=[5,5,int(64*3*f+C),C], stride=1, reuse=reuse, training=training, activation=tf.identity ) # h/2(192), C
        pred_flow = tf.image.resize_images( flow1, size=orisize )

        return pred_flow
