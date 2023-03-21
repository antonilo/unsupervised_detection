import os
from PIL import Image
import subprocess
import sys


# Parameters:  your video name
script_dir = os.path.dirname(os.path.abspath(__file__))
default_video_fname = script_dir + "/../download/video/todaiura_traffic.MOV"
video_fname = sys.argv[1] if len(sys.argv)>1 else default_video_fname


# create image files from a video
# fixed parameters
out_fps = "24" # same with DAVIS 2016 dataset.
out_width = 853
out_height = 480
img_rootdir = script_dir + "/../download/video/JPEGImages/480p/"

video_rootname, _ = os.path.splitext(os.path.basename(video_fname))
outimg_dir = img_rootdir + video_rootname # same directory architecture with DAVIS

if not os.path.exists(outimg_dir):
    os.makedirs(outimg_dir)

subprocess.call(['ffmpeg', '-i', video_fname, '-r', out_fps, '-vf', "scale=" + str(out_width) + ":" + str(out_height), outimg_dir+"/%05d.jpg"])


# Create empty black image as a fake annotatted mask
annotimg_dirname = script_dir + "/../download/video/Annotations/480p"

if not os.path.exists(annotimg_dirname):
    os.makedirs(annotimg_dirname)

empty_img = Image.new('RGB', (out_width, out_height), (0,0,0))
empty_img.save(annotimg_dirname + "/00000.png")


# Create image file text
imglist_dirname = script_dir + "/../download/video/ImageSets/480p"
imglist_fname = imglist_dirname + "/val.txt"
relative_img_dir = "/JPEGImages/480p/todaiura_traffic"
relative_annot_imgname = "/Annotations/480p/00000.png"

if not os.path.exists(imglist_dirname):
    os.makedirs(imglist_dirname)

fnames = sorted(os.listdir(outimg_dir))

with open(imglist_fname, 'w') as file:
    for fname in fnames:
        file.write(relative_img_dir + '/' + fname + ' ' + relative_annot_imgname + '\n')



