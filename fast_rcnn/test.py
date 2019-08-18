# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# Modified by CVML

"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
#from fast_rcnn.ftmap_transform import transformer_layer as trans_layer
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

#from nms.py_cpu_nms_rotated import py_cpu_nms

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

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
	#print ('lll: ', blobs['rois'])
    return blobs, im_scale_factors

def im_detect(net, im, boxes=None, extract_feat=False):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)
    #print 'blobs: ', blobs

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]
	#print ('lll: ', not cfg.TEST.HAS_RPN)

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        permanent_shape = im_blob.shape
	#print ('lll: ', permanent_shape)
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)
        print blobs['im_info']


    # reshape network inputs

    net.blobs['data'].reshape(*(blobs['data'].shape))

    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)

        
    #print('going to net forward')
    blobs_out = net.forward(**forward_kwargs)
    #print('blobs[rois]  : ', blobs_out)

    #print ('start_ind: ', list(net._layer_names))
    li = list(net._layer_names).index('roi_pool5')
    tops = [(net._blob_names[bi], net.blobs[net._blob_names[bi]].data.shape) for bi in list(net._top_ids(li))]
    #for bi in range(len(list(net._layer_names))):
	#print ('hello: ', net._blob_names[bi] ,net.blobs[net._blob_names[bi]].data.shape)
    '''print li
    print list(net._top_ids(li)), net._blob_names[28]
    print net.blobs['rois'].data.shape
    for ip in range (35):
	print ip, net._blob_names[ip]'''

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
	
	#print('rois :', rois[0,:])

	# unscale back to raw image space
        rpn_boxes = rois[:, 1:5] / im_scales[0]
	#print ('shape: ', rpn_boxes.shape)
	rpn_scores = net.blobs['scores'].data.copy()

	pred_scores = net.blobs['cls_prob'].data.copy()
	box_deltas = net.blobs['bbox_pred'].data.copy()
	#box_deltas = blobs_out['bbox_pred1']
        pred_boxes = bbox_transform_inv(rpn_boxes, box_deltas)
	#pred_boxes = np.hstack((rpn_boxes, rpn_boxes, rpn_boxes))
        pred_boxes = clip_boxes(pred_boxes, im.shape)

	#print('comp :', rpn_boxes[:50,:], pred_boxes[:50,:])
	orient_prob = net.blobs['orient_prob'].data.copy()

	warpedrois = net.blobs['warpedrois'].data.copy()
	transApplied = net.blobs['transApplied'].data.copy()
	rpn_boxes1 = warpedrois[:, 1:5] / im_scales[0]

        # use softmax estimated probabilities
        pred_scores1 = blobs_out['cls_prob1']
	#print 'im_detect'
	#print (pred_scores.shape)
        # Apply bounding-box regression deltas by accessing blob with name bbox_pred
        box_deltas1 = blobs_out['bbox_pred1']
        pred_boxes1 = bbox_transform_inv(rpn_boxes1, box_deltas1)
	#pred_boxes1 = np.hstack((rpn_boxes1, rpn_boxes1, rpn_boxes1))
        #pred_boxes1 = clip_boxes(pred_boxes1, im.shape)

        #orient_prob = blobs_out['orient_prob']
	#orient_prob = np.zeros((len(pred_scores),4), dtype='float')
        
	# unscale back to raw image space
        '''rpn_boxes = rois[:, 1:5] / im_scales[0]
	#print ('shape: ', rpn_boxes[45:60,:])
	rpn_scores = net.blobs['scores'].data.copy()

        # use softmax estimated probabilities
        pred_scores = blobs_out['cls_prob']
	#print 'im_detect'
	#print (pred_scores.shape)
        # Apply bounding-box regression deltas by accessing blob with name bbox_pred
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(rpn_boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)

        orient_prob = blobs_out['orient_prob']
	#orient_prob = np.zeros((len(pred_scores),4), dtype='float')

        if extract_feat == True:
        	conv_feat = net.blobs['conv5_3'].data.copy()
                #print conv_feat.shape
		return rpn_boxes, rpn_scores, pred_boxes, pred_scores, orient_prob, conv_feat, permanent_shape'''



    return rpn_boxes, rpn_scores, pred_boxes, pred_scores, orient_prob, pred_boxes1, pred_scores1, transApplied


def im_detect_new(net, im,  perm_shape, blobs, ross,im_scales, boxes=None, extract_feat=False):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

    #blobs, im_scales = _get_blobs(im, boxes)
    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        #print 'change me'
        #print im_blob.shape
        blobs['im_info'] = np.array(
            [[perm_shape[2], perm_shape[3], im_scales[0]]],
            dtype=np.float32)

	blobs['rois'] = np.array(ross)


    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
	#print 'rois shape', net.blobs['rois']
        net.blobs['rois'].reshape(*(blobs['rois'].shape))
	#print 'rois shape', net.blobs['rois'].shape
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}

    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
	forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)


    blobs_out = net.forward(**forward_kwargs)


    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
	# unscale back to raw image space
	#print 'in'
        rpn_boxes = rois[:, 1:5] / im_scales[0]
	#rpn_boxes = rois[:, 1:5]    
	#print ('shape: ', rpn_boxes)          
        #rpn_boxes = rois / im_scales[0]         
	#rpn_scores = net.blobs['scores'].data.copy()
	rpn_scores = np.array([[0.7]])
	#print ('shape1: ', rpn_scores)          

        # use softmax estimated probabilities
        pred_scores = blobs_out['cls_prob']

	#print ('shape2: ', pred_scores)          

        # Apply bounding-box regression deltas by accessing blob with name bbox_pred
        box_deltas = blobs_out['bbox_pred']
	#print 'box_deltas: ', (box_deltas.max())*16
        pred_boxes = bbox_transform_inv(rpn_boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
	#print ('pred_boxes: ', pred_boxes)          

        orient_prob = blobs_out['orient_prob']
	#orient_prob = np.zeros((len(pred_scores),4), dtype='float')

    	if extract_feat == True:
        	conv_feat = net.blobs['conv5_3'].data.copy()
                #print conv_feat.shape
		return rpn_boxes, rpn_scores, pred_boxes, pred_scores, orient_prob, conv_feat
        

    return rpn_boxes, rpn_scores, pred_boxes, pred_scores, orient_prob
    #return rpn_boxes, rpn_scores, pred_boxes

def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of boxes
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

def vis_detections_rpn(fname, class_name, dets, scores, im_name):
    """Visual debugging of detections."""

    for i in xrange(np.minimum(len(scores), dets.shape[0])):
        #print im_name
	im = cv2.imread(fname)
        bbox = map(int, dets[i, :])
	score = scores[i]

    	txt = str(score)
          
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), [0,0,255], 2, 16)
    	ret, baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    	cv2.rectangle(im, (bbox[0], bbox[1] - ret[1] - baseline),(bbox[0] + ret[0], bbox[1]), (255, 0, 0), -1)
        
    	cv2.putText(im, txt, (bbox[0], bbox[1] - baseline),
               	cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, 16)


	foldername = '/home/javed/1070/Projects/py-faster-rcnn-full_shift/data/output_rpn/' + im_name + '/' 
	if not os.path.isdir(foldername): 
    		os.makedirs(foldername)
	filename = foldername + str(i) + '.jpg'
        #print filename
        cv2.imwrite(filename, im)

#def vis_detections_final(im, class_name, all_final_boxes, im_name,thresh):
def vis_detections_final(im, class_name, all_final_boxes, im_name,thresh, cntG,cntR, cG, cR, rpn_sscores, rpn_bo, all_final_boxes_rotated):
    """Visual debugging of detections."""
    #print 'i am in visualizer'
    #print len(all_final_boxes)
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
                #print 'greater than thresh'
                #print bbox
    		txt = class_name + ': ' + str(orient_cls) + ': ' + str(score)
		#txt = class_name + ': ' + str(orient_cls) + ': ' + str(rpn_s)
		#print rpn_sscores
		s.append(score)
    		#cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), [0,0,255], 2, 16)
    		#cv2.rectangle(im, (rpn_bo[0], rpn_bo[1]), (rpn_bo[2], rpn_bo[3]), [255,0,255], 2, 16)
		#print('writing done')
    		#ret, baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    		#cv2.rectangle(im, (bbox[0], bbox[1] + ret[1] + baseline),
                # 		(bbox[0] + ret[0], bbox[1]), (255, 0, 0), -1)
        
    		#cv2.putText(im, txt, (bbox[0], bbox[1] + ret[1]+ baseline),
               	#	cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, 16)

		pts = np.array([[xAll[i,0],yAll[i,0]],[xAll[i,1],yAll[i,1]],[xAll[i,3],yAll[i,3]],[xAll[i,2],yAll[i,2]]], np.int32)
		#cv2.polylines(im, [pts],True,(0,255,255), 2)
		#cv2.polylines(im, [pts],True,(128,0,255), 2) #voilet like
		cv2.polylines(im, [pts],True,(147, 20,255), 6) # pink like

    if s:
#	print type(scores)
#	print 'max: ', max(scores)

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

def test_net(net, imdb, max_per_image=100, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    '''foldername = '/home/javed/1070/Projects/py-faster-rcnn-master/data/output_images_detected/' 
    foldername_all = '/home/javed/1070/Projects/py-faster-rcnn-master/data/output_images_all/'

    net2 = caffe.Net('/home/javed/1070/Projects/py-faster-rcnn-master/models/pascal_voc/VGG16/faster_rcnn_alt_opt/faster_rcnn_test_2.pt', '/home/javed/1070/Projects/py-faster-rcnn-master/output/faster_rcnn_alt_opt/voc_2007_trainval/VGG16_faster_rcnn_final.caffemodel', caffe.TEST) '''

    foldername = '/media/akhtar/6D2C8F896B2F79E0/Projects/py-faster-rcnn-master/data/output_images_detected/' 
    foldername_all = '/home/itu/faster-rcnn-1070/data/output_images_all/'

    '''net2 = caffe.Net('/home/itu/faster-rcnn-1070/models/pascal_voc/VGG16/faster_rcnn_alt_opt/faster_rcnn_test_3.pt', '/home/itu/faster-rcnn-1070/output/faster_rcnn_alt_opt/voc_2007_trainval/VGG16_faster_rcnn_final.caffemodel', caffe.TEST) '''

    all_boxes = [[] for _ in xrange(num_images)]

    ntopProp = [1,4,50,100,300]
    ntopProp = [300]
    #ntopProp = [50]
    theta = [0, 90, 135, 45, 157.5, 112.5, 67.5, 22.5]
    #theta = [45,90,135,45, 157.5, 112.5,67.5, 22.5]

    for t in xrange(0,len(ntopProp)):
    	output_dir = get_output_dir(imdb, net)

    	# timers
    	_t = {'im_detect' : Timer(), 'misc' : Timer()}

    	if not cfg.TEST.HAS_RPN:
        	roidb = imdb.roidb

    	all_final_boxes = [[[] for _ in xrange(num_images)]
                 	for _ in xrange(imdb.num_classes)]

	all_final_boxes_rotated = [[[] for _ in xrange(num_images)]
                 	for _ in xrange(imdb.num_classes)]

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
 
		fname = imdb.image_path_at(i)
		ind = fname.rindex('/')
		ind_end = fname.rindex('.')
		filename = fname[ind+1:ind_end]
                print filename

        	im = cv2.imread(imdb.image_path_at(i))

		fname = foldername + filename + '.jpg'

		'''rpn_boxes_rot = np.zeros((1,4))
                rpn_scores_rot = np.zeros((1))
                final_boxes_rot = np.zeros((1,12))
                final_scores_rot = np.zeros((1,3))
                orient_prob_rot = np.zeros((1,8))'''

        	_t['im_detect'].tic()
                #print 'first pass'
        	
                #rpn_boxes, rpn_scores, final_boxes, final_scores, orient_score, conv_feat, f_shape = im_detect(net, im, box_proposals, True)
                rpn_boxes, rpn_scores, final_boxes, final_scores, orient_score, final_boxes1, final_scores1, transApplied = im_detect(net, im, box_proposals, True)
                

		#print('orient_scores:  ', orient_score.shape)
                #print conv_feat.shape
                #in_feat = np.rollaxis(conv_feat, 1, 4)  
                #print in_feat.sum()

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

		#print('orient_scores:  ', orient_scores.shape)
		#top_proposals_pass_2 = 50
                temp_boxes = None
                blobs, im_scales = _get_blobs(im, temp_boxes)
                #print len(rpn_boxes)

		rotatedBoxesAll = np.zeros((len(rpn_boxes), 3,2,4))
		for iii in range(0, len(rpn_boxes)):
			final_boxes_tr = final_boxes1[iii,:]
			#print('final_boxes_tr :', final_boxes_tr)
			final_boxes_tr = ((final_boxes_tr * im_scales[0]) / 16)

			final_boxes_tr = trans_box1(final_boxes_tr,transApplied[iii,0,:,:],transApplied[iii,1,:,:])
			
			final_boxes_tr = ((final_boxes_tr * 16) / im_scales[0])

			rotatedBoxesAll[iii, :,:,:] = final_boxes_tr[0,:,:,:]
			

		

                #print('rotatedBoxesAll :', rotatedBoxesAll.shape)  
        	#vis_detections_rpn(fname, 'fireArm', rpn_boxes, rpn_scores, filename)
                #print hi

   		rpn_dets = np.hstack((rpn_boxes, rpn_scores)) \
                	.astype(np.float32, copy=False)
		all_rpn_boxes[0][i] = rpn_dets


        	_t['misc'].tic()

        	# skip j = 0, because it's the background class
		#maxScore = np.maximum(final_scores1, final_scores)
		maxScore = final_scores1
        	for j in xrange(1, imdb.num_classes):
			#inds = np.where(final_scores1[:, j] > thresh)[0]
			#cls_scores = final_scores1[inds, j]
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

			'''keep = py_cpu_nms(cls_dets_temp_rotated, 0.4)
			cls_dets = cls_dets[keep, :]
			rotatedBoxesClass = rotatedBoxesClass[keep, :]
			cls_dets_temp = cls_dets_temp[keep,:]'''

            		keep = nms(cls_dets_temp, cfg.TEST.NMS)
			#keep = nms(cls_dets_temp, 0.3)
			
			cls_dets = cls_dets[keep, :]
			rotatedBoxesClass = rotatedBoxesClass[keep, :]
			
			'''cls_dets_temp_rotated = cls_dets_temp_rotated[keep,:]
			keep = py_cpu_nms(cls_dets_temp_rotated, 0.4)
			#print('keep :', keep)
            		cls_dets = cls_dets[keep, :]
			rotatedBoxesClass = rotatedBoxesClass[keep, :]'''

            		all_final_boxes[j][i] = cls_dets
			all_final_boxes_rotated[j][i] = rotatedBoxesClass
			#print('rotatedBoxesClass :', rotatedBoxesClass.shape, cls_dets.shape)

		#print('all_final_boxes_rotated :', all_final_boxes_rotated)
		#print('all_final_boxes :', all_final_boxes)
        	# Limit to max_per_image detections *over all classes*

        	if max_per_image > 0:
            		image_scores = np.hstack([all_final_boxes[j][i][:, 4]
                          		            for j in xrange(1, imdb.num_classes)])

            		if len(image_scores) > max_per_image:
                		image_thresh = np.sort(image_scores)[-max_per_image]
                		for j in xrange(1, imdb.num_classes):
                   			keep = np.where(all_final_boxes[j][i][:, -1] >= image_thresh)[0]
                    			all_final_boxes[j][i] = all_final_boxes[j][i][keep, :]
					all_final_boxes_rotated[j][i] = all_final_boxes_rotated[j][i][keep, :]

                
                for j in xrange(1, imdb.num_classes):
			#rpn_bo = np.array([616, 405, 825, 556])
			#rpn_bo = np.array([231,129,621,939])
			rpn_bo = np.array([208, 58, 2243, 1094])
			#im = vis_detections_final(im, imdb.classes[j], all_final_boxes[j][i], filename, 0.65)
			im,cntG,cntR, cG, cR = vis_detections_final(im, imdb.classes[j], all_final_boxes[j][i], filename, 0.75, cntG,cntR, cG, cR, rpn_sscores, rpn_bo, all_final_boxes_rotated[j][i])
                        #print hi

		fname = foldername_all + filename + '.jpg'
                print fname
		cv2.imwrite(fname, im)
                    

        	_t['misc'].toc()
        	print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              	.format(i + 1, num_images, _t['im_detect'].average_time,
                      	_t['misc'].average_time)


		#print('all_rpn_boxes', len(all_rpn_boxes), len(all_final_boxes), output_dir)
    	#print ('Evaluating RPN detections for top Proposals: ' + str(ntopProp[t]) )
    	#imdb.evaluate_rpn(all_rpn_boxes, output_dir, ntopProp[t])

    	#print ('Evaluating detections for top Proposals: ' + str(ntopProp[t]) )
    	#imdb.evaluate_detections(all_final_boxes, output_dir,ntopProp[t])


def trans_box1(final_boxes,T_final, T11):
    final_boxes = final_boxes.reshape(1,12)
    final_boxes_final = np.zeros((len(final_boxes),3, 2,4))
    #print('final_boxes :', final_boxes.shape, final_boxes_final.shape)

    for k in range(0, len(final_boxes)):
        #print('k :', k)
	
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
        #final_boxes_final[k,:] = [ class1_out[0], class1_out[1], class1_out[2], class1_out[3], class2_out[0], class2_out[1], class2_out[2], class2_out[3], class3_out[0], class3_out[1], class3_out[2], class3_out[3]]

    return final_boxes_final

def trans_layer1(T_final,T11, final_b):

	nT0 = inv(T11)
	ncorner_pts = [[final_b[0],final_b[2],final_b[0],final_b[2]],[final_b[1],final_b[1],final_b[3],final_b[3]],[1,1,1,1]]
	nboxx = np.dot(nT0[0:2,:],ncorner_pts)
	rxymin_nb = nboxx.min(1)
	rxymax_nb = nboxx.max(1)

	T2 = inv(T_final)
	boxx2 = np.dot(T2[0:2,:],[nboxx[0], nboxx[1],[1,1,1,1]])

	#print('nboxx', nboxx.shape, boxx2.shape)
	
	#fin_cropped_box =  [rxymin_nb[0], rxymin_nb[1], rxymin_nb[0], rxymin_nb[1]]


    	
	return boxx2
