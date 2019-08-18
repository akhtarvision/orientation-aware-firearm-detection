#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------


# Modified by CVML group @ITU- Punjab
"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from utils.blob import im_list_to_blob
from numpy.linalg import inv

CLASSES = ('__background__',
           'Gun','Riffle')

NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_cascade_firearms_iter_60000.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}



def vis_detections_final(im, class_name, all_final_boxes,thresh, cntG,cntR, cG, cR, rpn_sscores, rpn_bo, all_final_boxes_rotated):
    """Visual debugging of detections."""
    #print 'i am in visualizer'
    #print len(all_final_boxes)

    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im, aspect='equal')


    boxes = all_final_boxes[:,:4]
    scores = all_final_boxes[:,4]
    scor = all_final_boxes[:,10]  
    rpnns =   all_final_boxes[:,6:10] 

    xAll = all_final_boxes_rotated[:,:4]
    yAll = all_final_boxes_rotated[:,4:8]

    orient_class = all_final_boxes[:,5]
    s=[]
    for i in xrange(len(scores)):
    	
	bbox = map(int, boxes[i,:])
	#rpn_bo = map(int, rpnns[i,:])
	score = scores[i]
	orient_cls = orient_class[i]
	rpn_s = scor[i]


	if score > thresh:

    		txt = class_name + ': ' + str(orient_cls) + ': ' + str(score)

		s.append(score)

		pts = np.array([[xAll[i,0],yAll[i,0]],[xAll[i,1],yAll[i,1]],[xAll[i,3],yAll[i,3]],[xAll[i,2],yAll[i,2]]], np.int32)
		#cv2.polylines(im, [pts],True,(0,255,255), 2)
		#cv2.polylines(im, [pts],True,(128,0,255), 2) #voilet like
		cv2.polylines(im, [pts],True,(147, 20,255), 6) # pink like

    if s:


	    if (class_name == 'Gun'):
		cntG = max(s)+cntG
		cG=cG+1

	    if (class_name == 'Riffle'):
		cntR = max(s)+cntR
		cR=cR+1

    #print (cntG,cntR)
    #print (cG,cR)

    return im,cntG,cntR, cG, cR
    #return im



def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    #if not cfg.TEST.HAS_RPN:
        #blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
	#print ('lll: ', blobs['rois'])
    return blobs, im_scale_factors

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    num_images = 1
    foldername = '/media/akhtar/6D2C8F896B2F79E0/Projects/py-faster-rcnn-master/data/output_images_detected/' 
    foldername_all = '/home/itu/faster-rcnn-1070/data/output_images_all/'
    thresh=0.05
    max_per_image=100

    all_boxes = [[] for _ in xrange(num_images)]

    ntopProp = [300]


    theta = [0, 90, 135, 45, 157.5, 112.5, 67.5, 22.5]

    for t in xrange(0,len(ntopProp)):
    	#output_dir = get_output_dir(imdb, net)



    	if not cfg.TEST.HAS_RPN:
        	roidb = imdb.roidb

    	all_final_boxes = [[[] for _ in xrange(num_images)]
                 	for _ in xrange(3)]

	all_final_boxes_rotated = [[[] for _ in xrange(num_images)]
                 	for _ in xrange(3)]

    	all_rpn_boxes = [[[] for _ in xrange(num_images)]
                 	for _ in xrange(1)]

	#print('all_final_boxes_rotated :', all_final_boxes_rotated)
	cntG = 0
	cntR = 0
	cG = 0
	cR = 0

    	for i in xrange(num_images):
		# filter out any ground truth boxes
		if cfg.TEST.HAS_RPN:
	    		box_proposals = None
		else:
	    	# The roidb may contain ground-truth rois (for example, if the roidb
	    	# comes from the training or val split). We only want to evaluate
	    	# detection on the *non*-ground-truth rois. We select those the rois
	    	# that have the gt_classes field set to 0, which means there's no
	    	# ground truth.
	    		box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]



	    # Load the demo image
		im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
	
		im = cv2.imread(im_file)




		rpn_boxes, rpn_scores, final_boxes, final_scores, orient_score, final_boxes1, final_scores1, transApplied = im_detect(net, im, box_proposals, True)

		if ntopProp[t] == 300:
			if len(rpn_scores) > 299:
				rpn_boxes = rpn_boxes[0:ntopProp[t],:]
				rpn_scores = rpn_scores[0:ntopProp[t],:]
				final_boxes = final_boxes[0:ntopProp[t],:]
				final_scores = final_scores[0:ntopProp[t],:]
				orient_scores = orient_score[0:ntopProp[t],:]
				final_boxes1 = final_boxes1[0:ntopProp[t],:]
				final_scores1 = final_scores1[0:ntopProp[t],:]
				transApplied = transApplied[0:ntopProp[t],:,:,:]
		else:	
			rpn_boxes = rpn_boxes[0:ntopProp[t],:]
			rpn_scores = rpn_scores[0:ntopProp[t],:]
			final_boxes = final_boxes[0:ntopProp[t],:]
			final_scores = final_scores[0:ntopProp[t],:]
			orient_scores = orient_score[0:ntopProp[t],:]
			final_boxes1 = final_boxes1[0:ntopProp[t],:]
			final_scores1 = final_scores1[0:ntopProp[t],:]
			transApplied = transApplied[0:ntopProp[t],:,:,:]

		temp_boxes = None
		blobs, im_scales = _get_blobs(im, temp_boxes)

		rotatedBoxesAll = np.zeros((len(rpn_boxes), 3,2,4))

		for iii in range(0, len(rpn_boxes)):
			final_boxes_tr = final_boxes1[iii,:]
			#print('final_boxes_tr :', final_boxes_tr)
			final_boxes_tr = ((final_boxes_tr * im_scales[0]) / 16)

			final_boxes_tr = trans_box1(final_boxes_tr,transApplied[iii,0,:,:],transApplied[iii,1,:,:])
	
			final_boxes_tr = ((final_boxes_tr * 16) / im_scales[0])

			rotatedBoxesAll[iii, :,:,:] = final_boxes_tr[0,:,:,:]
	

		rpn_dets = np.hstack((rpn_boxes, rpn_scores)) \
			.astype(np.float32, copy=False)
		#all_rpn_boxes[0][i] = rpn_dets


		#_t['misc'].tic()

		maxScore = final_scores1
		for j in xrange(1, 3):

			inds = np.where(maxScore[:, j] > thresh)[0]
			cls_scores = maxScore[inds, j]
			cls_boxes = final_boxes[inds, j*4:(j+1)*4]
			cls_orient = np.argmax(orient_score[inds, :], axis = 1)
			rpn_bboxes = rpn_boxes[inds,:]
			rpn_sscores = rpn_scores[inds]
	
			cls_scores1 = final_scores[inds, j]

			rotatedBoxesClass = np.hstack((rotatedBoxesAll[inds,j,0,:], rotatedBoxesAll[inds,j,1,:])).astype(np.float32, copy=False)
			#print('rotatedBoxesClass :', rotatedBoxesClass.shape)

			cls_dets_temp_rotated = np.hstack((rotatedBoxesAll[inds,j,0,:], rotatedBoxesAll[inds,j,1,:], cls_scores[:, np.newaxis])) \
				.astype(np.float32, copy=False)


	    		cls_dets_temp = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
				.astype(np.float32, copy=False)

			#print('cls_dets_temp', cls_dets_temp.shape)

	   		cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis], cls_orient[:, np.newaxis], rpn_bboxes, rpn_sscores)) \
				.astype(np.float32, copy=False)


	    		keep = nms(cls_dets_temp, cfg.TEST.NMS)
			#keep = nms(cls_dets_temp, 0.3)
	
			cls_dets = cls_dets[keep, :]
			rotatedBoxesClass = rotatedBoxesClass[keep, :]
	

	    		all_final_boxes[j][i] = cls_dets
			all_final_boxes_rotated[j][i] = rotatedBoxesClass

		if max_per_image > 0:
	    		image_scores = np.hstack([all_final_boxes[j][i][:, 4]
			  		            for j in xrange(1, 3)])

	    		if len(image_scores) > max_per_image:
				image_thresh = np.sort(image_scores)[-max_per_image]
				for j in xrange(1, 3):
		   			keep = np.where(all_final_boxes[j][i][:, -1] >= image_thresh)[0]
		    			all_final_boxes[j][i] = all_final_boxes[j][i][keep, :]
					all_final_boxes_rotated[j][i] = all_final_boxes_rotated[j][i][keep, :]

	
		for j in xrange(1, 3):

			rpn_bo = np.array([208, 58, 2243, 1094])


			im,cntG,cntR, cG, cR = vis_detections_final(im, CLASSES[j], all_final_boxes[j][i], 0.75, cntG,cntR, cG, cR, rpn_sscores, rpn_bo, all_final_boxes_rotated[j][i])


		print ('check: ',os.path.join(cfg.DATA_DIR, 'demo', 're_'+image_name))
		cv2.imwrite(os.path.join(cfg.DATA_DIR, 'demo', 're_'+image_name), im)
		    


def trans_box1(final_boxes,T_final, T11):
	final_boxes = final_boxes.reshape(1,12)
	final_boxes_final = np.zeros((len(final_boxes),3, 2,4))

	for k in range(0, len(final_boxes)):

		class1 = final_boxes[k,0:4]
		class2 = final_boxes[k,4:8]
		class3 = final_boxes[k,8:12]

		box1 = [ class1[0] , class1[1] , class1[2] , class1[3] ]
		box2 = [ class2[0] , class2[1] , class2[2] , class2[3] ] 
		box3 = [ class3[0] , class3[1] , class3[2] , class3[3] ]

		class1_out = trans_layer1(T_final, T11, box1)
		class2_out = trans_layer1(T_final, T11, box2)
		class3_out = trans_layer1(T_final, T11, box3)

		final_boxes_final[k,0,:,:] = class1_out
		final_boxes_final[k,1,:,:] = class2_out
		final_boxes_final[k,2,:,:] = class2_out

		return final_boxes_final

def trans_layer1(T_final,T11, final_b):

	nT0 = inv(T11)
	ncorner_pts = [[final_b[0],final_b[2],final_b[0],final_b[2]],[final_b[1],final_b[1],final_b[3],final_b[3]],[1,1,1,1]]
	nboxx = np.dot(nT0[0:2,:],ncorner_pts)
	rxymin_nb = nboxx.min(1)
	rxymax_nb = nboxx.max(1)

	T2 = inv(T_final)
	boxx2 = np.dot(T2[0:2,:],[nboxx[0], nboxx[1],[1,1,1,1]])

	return boxx2


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    print ('prototxt: ',prototxt)
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])
    print ('caffemodel: ',caffemodel)

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)


    im_names = ['north+korea+army_38.png']
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)
