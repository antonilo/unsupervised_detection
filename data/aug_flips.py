import tensorflow as tf

def left_right( img ):
    img = img[:,::-1,:]
    return img

def top_down(img):
    img = img[::-1,:,:]
    return img

def rotate_180_img(img1):
    img1 = left_right(img1)
    img1 = top_down(img1)
    return img1


def keep_rotate(img1, img2):
    cases = tf.constant(2) # Nothing, rotate_180
    current_case = tf.random_uniform([], minval=0, maxval=cases, dtype=tf.int32)

    img1, img2 = tf.cond(tf.equal(current_case, 0), lambda: (img1, img2),
                         lambda: (rotate_180_img(img1), rotate_180_img(img2)))
    return img1, img2


def vert_hor_flip(img1, img2):
    cases = tf.constant(2) # Left right, top_down
    current_case = tf.random_uniform([], minval=0, maxval=cases, dtype=tf.int32)
    img1, img2 = tf.cond(tf.equal(current_case, 0),
                         lambda: (left_right(img1), left_right(img2)),
                         lambda: (top_down(img1), top_down(img2)))
    return img1, img2


def random_flip_images(img1, img2):
    """
    This function will make a random flipping of the images.
    There is 50% probability of flipping on each axis.
    """
    cases = tf.constant(2) # Nothing, left_right, top_down, rotate_180
    current_case = tf.random_uniform([], minval=0, maxval=cases, dtype=tf.int32)

    img1, img2 = tf.cond(tf.equal(current_case, 0), lambda: keep_rotate(img1, img2),
                         lambda: vert_hor_flip(img1, img2))
    return img1, img2
