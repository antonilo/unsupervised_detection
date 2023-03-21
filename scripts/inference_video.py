#!/usr/bin/env python
# coding: utf-8

# In[1]:


import subprocess
import os
import glob


# ## create image files from a video

# In[36]:


video_name = "../download/video/todaiura_traffic.MOV"
outimg_dir = "../download/video/JPEGImages/480p/todaiura_traffic"
out_fps = "24"
out_height = "480"


# In[37]:


subprocess.call(['ffmpeg', '-i', video_name, '-r', out_fps, '-vf', "scale=-1:"+out_height, outimg_dir+"/%05d.jpg"])


# ## create image file text

# In[41]:


imglist_fname = "../download/video/ImageSets/480p/val.txt"
relative_img_dir = "/JPEGImages/480p/todaiura_traffic"
relative_annot_imgname = "/Annotations/480p/00000.png" # I brought this from DAVIS bear data.


# In[42]:


fnames = os.listdir(outimg_dir)


# In[45]:


with open(imglist_fname, 'w') as file:
    for fname in fnames:
        file.write(relative_img_dir + '/' + fname + ' ' + relative_annot_imgname + '\n')


# In[44]:





# In[ ]:




