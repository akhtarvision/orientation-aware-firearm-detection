# --------------------------------------------------------
# Written by CVML
# --------------------------------------------------------



from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import argparse
from utils.timer import Timer
import numpy as np
import cv2
from numpy.linalg import inv
import caffe
from utils.blob import im_list_to_blob
import os
import matplotlib.pyplot as plt
import gc
import yaml


DEBUG = False

class makebBox_regionProposal(caffe.Layer):
    """
    transforms feature map and corresponding bouning boxes with respect to angle
    """

    def setup(self, bottom, top):

	pass

    def forward(self, bottom, top):


        #assert bottom[0].data.shape[0] == 1, \
        #    'Only single item batches are supported'


        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        rpn_boxes_Actual = bottom[0].data
        box_deltas = bottom[1].data
	im_info = bottom[2].data
	scores = bottom[3].data


	#print('im_info : ', im_info)

	im_scales = im_info[0][2]

	im_shape = np.array([im_info[0][0], im_info[0][1]]) / im_scales
	#print('conv_feat : ', conv_feat.shape)
	#print('rpnBoxes : ', rpn_boxes.shape)
	

	#for idx in range(len(cls_idx)):
		#cls_boxes = final_boxes[inds, j*4:(j+1)*4]


	rpn_boxes = rpn_boxes_Actual[:, 1:5] / im_scales
	pred_boxes = bbox_transform_inv(rpn_boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im_shape)

	cls_idx = np.argmax(scores, axis = 1)

	#print('cls_idx', cls_idx.shape, cls_idx )

	#cls_idx = cls_idx.reshape(len(cls_idx), 1)
	#print('cls_idx', cls_idx.shape)
	#pred_boxes = pred_boxes[:, cls_idx*4:(cls_idx+1)*4]
	temp = np.zeros((len(cls_idx), 5))

	for idx in range(len(cls_idx)):
		#print(cls_idx[idx])
		temp[idx,1:] = pred_boxes[idx, cls_idx[idx]*4:(cls_idx[idx]+1)*4]

	
	pred_boxes = temp * im_scales
	#addd = cls_idx >0
	#print('Compare :', rpn_boxes[cls_idx>0,:], temp[cls_idx>0,:])

	#rpn_boxes_Actual[:,1:5] = pred_boxes

	top[0].reshape(*(pred_boxes.shape))
        top[0].data[...] = pred_boxes

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
	top[0].reshape(*bottom[0].shape)
	#top[1].reshape(*bottom[1].shape)
        #pass




