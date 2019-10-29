# --------------------------------------------------------
# Written by CVML
# --------------------------------------------------------



from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from fast_rcnn.ftmap_transform import transformer_layer_fMapSep as fMap_trans_layer
from fast_rcnn.ftmap_transform import transformer_layer_fMapSep_backward as fMap_trans_layer_backward
import argparse
from utils.timer import Timer
import numpy as np
import cv2
from numpy.linalg import inv
import caffe
from fast_rcnn.nms_wrapper import nms
import cPickle
from utils.blob import im_list_to_blob
import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import gc
import yaml


DEBUG = False

class fMapWarpLayerSep(caffe.Layer):
    """
    transforms feature map and corresponding bouning boxes with respect to angle
    """

    def setup(self, bottom, top):

	pass

    def forward(self, bottom, top):

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'


        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        conv_feat = bottom[0].data
        rpn_boxes = bottom[1].data
	angle = bottom[2].data

	out_feat, ross, transApplied, T_final = warp_fMap(conv_feat, rpn_boxes, angle) 

        blob = np.rollaxis(out_feat, 3, 1) 

	top[0].reshape(*(blob.shape))
        top[0].data[...] = blob
	top[1].reshape(*(ross.shape))
	top[1].data[...] = ross
	
	top[2].reshape(*(transApplied.shape))
	top[2].data[...] = transApplied
	
	#top[3].reshape(*(T_final.shape))
	#top[3].data[...] = T_final

	#print('blob', blob.shape, ross.shape)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""

        grad_warpMap = top[0].diff
        #rpn_boxes = bottom[1].data
        rpn_boxes_gwm = top[1].data
	angle = bottom[2].data

	in_gwm = np.rollaxis(grad_warpMap, 1, 4)

	out_gwm, rotated_gwm, transApplied_gwm = fMap_trans_layer_backward(in_gwm, angle, rpn_boxes_gwm)


    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
	top[0].reshape(*bottom[0].shape)
	top[1].reshape(*bottom[1].shape)
	top[2].reshape(*bottom[2].shape)
        #pass



def warp_fMap(conv_feat, rpn_boxes, angle):

	
	#angle = 22.5
        
        #print conv_feat.shape
        in_feat = np.rollaxis(conv_feat, 1, 4)  
        #print in_feat.sum()


	top_proposals_pass_2 = 50
        temp_boxes = None
		
	out_feat, rotated_rpns, transApplied, T_final = fMap_trans_layer(in_feat, angle, rpn_boxes)

	ross = rotated_rpns

	ross = np.array(ross)
	#print('ross :', ross.shape)
	
	transApplied = np.array(transApplied)

	return out_feat, ross, transApplied, T_final



