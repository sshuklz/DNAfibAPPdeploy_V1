#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:00:28 2022

@author: Shalabh
"""

import numpy as np
import cv2
import base64
import plotly.graph_objects as go

def blank_fig():
    
    fig = go.Figure(go.Scatter(x=[], y = []))
    fig.update_layout(template = None)
    fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
    fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)
    
    return fig

def read_image_string(contents):
    
   encoded_data = contents[0].split(',')[1]
   nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   
   return img

def parse_contents(contents, filename, date):
    
    image_mat = read_image_string(contents=contents)
    
    return image_mat

class ImageOperations(object):
    
    def __init__(self, image_file_src):
        
        self.image_file_src = image_file_src
        self.MAX_PIXEL = 255
        self.MIN_PIXEL = 0
        self.MID_PIXEL = self.MAX_PIXEL // 2
    
    def read_operation(self):
        
        image_src = self.image_file_src
            
        return image_src
    
    def crop_operation(self, x0, x1, y0, y1):
        
        image_src = self.image_file_src
        image_src = image_src[y0:y1 , x0:x1]  
        
        return image_src
        
    def G_R_operation(self, x0, x1, y0, y1):
        
        image_src = self.image_file_src
        image_src = image_src[y0:y1 , x0:x1]  
        reds = sum(image_src[:,:,0].flatten())
        greens = sum(image_src[:,:,1].flatten())
        
        if reds > greens:
        
            reds_new = reds / reds
            greens_new = greens / reds
            
        else:
            
            reds_new = reds / greens
            greens_new = greens / greens
            
        ratio =  ("%.2f" % greens_new) + ' : ' + ("%.2f" % reds_new)
        
        return ratio

    def gamma_operation(self, thresh_val):
        
        image_src = self.image_file_src
        invGamma = 1 / (thresh_val)
 
        table = [((i / 255) ** invGamma) * 255 for i in range(256)]
        table = np.array(table, np.uint8)
        
        image_src = cv2.LUT(image_src, table)
        return image_src
    
    def denoiseI_operation(self, thresh_val):
        
        image_src = self.image_file_src
        image_src = cv2.fastNlMeansDenoisingColored(image_src,thresh_val,thresh_val,0,7,21)
        
        return image_src
    
    def contrast_operation(self, thresh_val):
        
        image_src = self.image_file_src
        image_src = cv2.convertScaleAbs(image_src, alpha=thresh_val, beta=0)

        return image_src

    def CR_operation(self, thresh_val):
        
        image_src = self.read_operation()
        image_src[:,:,0][image_src[:,:,0] < thresh_val] = 0
        
        return image_src
        
    def GR_operation(self, thresh_val):
        
        image_src = self.read_operation()
        image_src[:,:,1][image_src[:,:,1] < thresh_val] = 0
        
        return image_src
    
    def BR_operation(self, thresh_val):
        
        image_src = self.read_operation()
        image_src[:,:,2][image_src[:,:,2] < thresh_val] = 0
        
        return image_src