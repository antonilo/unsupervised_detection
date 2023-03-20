#!/usr/bin/env python
# coding: utf-8

# In[11]:


import subprocess
import os
import glob


# ## create image files from a video

# In[9]:


video_name = "../download/video/todaiura_traffic.MOV"
out_fps = "24"
out_dir = "../download/video/JPEGImages/480p"


# In[10]:


subprocess.call(['ffmpeg', '-i', video_name, '-r', out_fps, out_dir+"/%05d.jpg"])


# ## create image file text

# In[ ]:




