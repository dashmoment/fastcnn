#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:12:51 2017

@author: ubuntu
"""
import imghdr
import os
from PIL import Image

def purgeinvalid_img(directory, ftype=".jpg", validtype = "jpeg"):
    
   
    
    for file in os.listdir(directory):
        #print(file)
        if file.endswith(ftype):
            
            filename = os.path.join(directory, file)
            filetype = imghdr.what(filename)  
            
            if filetype != validtype:
                os.remove(filename)
                print("Remove{}".format(filename))
                
           
    print("Directory Purged")

#purgeinvalid_img("./ilsvrc11")

def test():
     ftype = imghdr.what("../ilsvrc11/n00005787_12948.jpg")  
     print(ftype)
     
def purgeinvalidandRGB_img(directory, ftype=".jpg", validtype = "jpeg"):
    
   
    
    for file in os.listdir(directory):
        #print(file)
        if file.endswith(ftype):
            filename = os.path.join(directory, file)
            filetype = imghdr.what(filename)  
            if filetype != validtype:
                os.remove(filename)
                print("Remove{}".format(filename))
                continue
            
            im = Image.open(filename)         
            
            
            if im.mode != 'RGB':
                im = im.convert('RGB')
                #im.show()
                im.save(filename)
                print(im.mode)
                continue
            
           
           

                
           
    print("Directory Purged")
     
