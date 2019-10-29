# Modified & Written by CVML

import numpy as np
import matplotlib.pyplot as plt
import cv2
import re
import math
from numpy.linalg import inv
from numpy import linalg
from numpy import matrix
import xml.etree.ElementTree as ET

def transformer_layer(input_fmap, angle, box, output_tr_flag, out_dims=None,  **kwargs):
    
  
    B = np.shape(input_fmap)[0]
    H = np.shape(input_fmap)[1]
    W = np.shape(input_fmap)[2]
    C = np.shape(input_fmap)[3]
    
   
    cntr = np.asarray([(box[1]+box[3])/2, (box[0]+box[2])/2])

    T1 = [[1,0,0],[0,1,0],[-cntr[1],-cntr[0],1]]
    T2 = np.asarray([[np.cos(np.deg2rad(int(angle))), np.sin(np.deg2rad(int(angle))), 0],[-np.sin(np.deg2rad(int(angle))),np.cos(np.deg2rad(int(angle))),0],[0,0,1]])
    T3 = [[1,0,0],[0,1,0],[cntr[1],cntr[0],1]]

    T = np.dot(np.transpose(T3),np.dot(np.transpose(T2),np.transpose(T1)))
  
    corner_pts = [[0,W-1,0,W-1],[0,0,H-1,H-1],[1,1,1,1]]

    trans_cpoints = np.dot(T[0:2,:],corner_pts)
    
    xymin = trans_cpoints.min(1)
    xymax = trans_cpoints.max(1)

    out_H = np.int32(xymax[1] - xymin[1] + 1)
    out_W = np.int32(xymax[0] - xymin[0] + 1)

    out_fmap_size = [0,0,out_W, out_H]

    T4 = [[1,0,0],[0,1,0],[-xymin[0], -xymin[1],1]];
   
    T_final = np.dot(np.transpose(T4),T);

    rect_pts = [ [box[0],box[2], box[0], box[2]] ,[box[1],box[1],box[3],box[3]],[1,1,1,1]]

    trans_rpoints = np.dot(T_final[0:2,:],rect_pts)

    rxymin = trans_rpoints.min(1)
    rxymax = trans_rpoints.max(1)

    cropped_box = [np.int32(np.floor(rxymin[0])), np.int32(np.floor(rxymin[1])), np.int32(np.floor(rxymax[0])), np.int32(np.floor(rxymax[1]))]

    intSec1, intSec2 = LinesIntersectionForLargestBox(trans_rpoints, np.array(rect_pts), angle)
    height_deltas = [intSec1[1]-cropped_box[1], cropped_box[3]-intSec2[1]]

    if output_tr_flag == True:
        
        return cropped_box
    
    else:
        
    	batch_grids = affine_grid_generator(out_H, out_W, T_final)
    
    	x_s = batch_grids[:,0, :, :]
    	y_s = batch_grids[:,1, :, :]

	out_fmap = bilinear_sampler_Interpol(input_fmap, x_s, y_s) # Interpolation with in bbox and extend outside using Ia
    
    	negative_flag = False
    	if cropped_box[0] < 0:

       		cropped_box[2] = int(cropped_box[2]) - int(cropped_box[0])
       		cropped_box[0] = int(cropped_box[0]) - int(cropped_box[0])

    	if cropped_box[1] < 0:

       		cropped_box[3] = int(cropped_box[3]) - int(cropped_box[1])
       		cropped_box[1] = int(cropped_box[1]) - int(cropped_box[1])


    	f_map = out_fmap[:, cropped_box[1]:cropped_box[3] , cropped_box[0]:cropped_box[2] ,:]

    	return f_map, T_final, cropped_box, negative_flag, trans_rpoints, trans_cpoints, out_fmap_size, height_deltas


def affine_grid_generator(H, W, theta):
 

    # create normalized 2D grid
    x = np.arange(W)
    y = np.arange(H)

    x_t, y_t = np.meshgrid(x, y)
    
    # flatten
    x_t_flat = np.reshape(x_t, (-1))
    y_t_flat = np.reshape(y_t, (-1))
         
    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = np.ones((np.shape(x_t_flat)[0])) 

    sampling_grid = np.stack([x_t_flat, y_t_flat, ones])

    # transform the sampling grid - batch multiply
    theta_inv = np.linalg.inv(theta)

    out_sampGrid = np.dot(theta_inv[0:2,:], sampling_grid)    
    
    # batch grid has shape (num_batch, 2, H*W)
    # reshape to (num_batch, 2, H, W)
    batch_grids = out_sampGrid.reshape((1, 2, H, W))
        
    return batch_grids

def bilinear_sampler(input_fmap, x, y):
    
    # prepare useful params
    B = np.shape(input_fmap)[0]
    H = np.shape(input_fmap)[1]
    W = np.shape(input_fmap)[2]
    C = np.shape(input_fmap)[3]

    max_y = np.int32(H - 1)
    max_x = np.int32(W - 1)
    
    zero = np.zeros([], dtype='int32')
    
    # grab 4 nearest corner points for each (x_i, y_i)
    # i.e. we need a rectangle around the point of interest
    x0 = np.int32(np.floor(x))
    x1 = np.int32(x0 + 1)
    y0 = np.int32(np.floor(y))
    y1 = np.int32(y0 + 1)
    

    # clip to range [0, H/W] to not violate img boundaries
    x0 = np.clip(x0, zero, max_x)
    x1 = np.clip(x1, zero, max_x)
    y0 = np.clip(y0, zero, max_y)
    y1 = np.clip(y1, zero, max_y) 
    
    # get pixel value at corner coords
    Ia = input_fmap[0,y0,x0,:]
    Ib = input_fmap[0,y1,x0,:]
    Ic = input_fmap[0,y0,x1,:]
    Id = input_fmap[0,y1,x1,:]

    # recast as float for delta calculation
    x0 = np.float32(x0)
    x1 = np.float32(x1)
    y0 = np.float32(y0)
    y1 = np.float32(y1)

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)


    # add dimension for addition
    wa = np.expand_dims(wa,axis=3)
    wb = np.expand_dims(wb,axis=3)
    wc = np.expand_dims(wc,axis=3)
    wd = np.expand_dims(wd,axis=3)
    
    wa[np.where(wa<0)]=0
    wb[np.where(wb<0)]=0
    wc[np.where(wc<0)]=0
    wd[np.where(wd<0)]=0
    
    output_fmap = Ib

    return output_fmap

def bilinear_sampler_Interpol(input_fmap, x, y):
    
    # prepare useful params
    B = np.shape(input_fmap)[0]
    H = np.shape(input_fmap)[1]
    W = np.shape(input_fmap)[2]
    C = np.shape(input_fmap)[3]

    max_y = np.int32(H - 1)
    max_x = np.int32(W - 1)
    
    zero = np.zeros([], dtype='int32')
    
    # grab 4 nearest corner points for each (x_i, y_i)
    # i.e. we need a rectangle around the point of interest
    x0 = np.int32(np.floor(x))
    x1 = np.int32(x0 + 1)
    y0 = np.int32(np.floor(y))
    y1 = np.int32(y0 + 1)
    

    ad1 = 1*(x0>=0)
    ad2 = 1*(x0<=max_x)
    ad3 = 1*(y0>=0)
    ad4 = 1*(y0<=max_y)

    maskx = 1*(ad1[0,:,:] * ad2[0,:,:])
    masky = 1*(ad3[0,:,:] * ad4[0,:,:])
    mask = maskx*masky
    mask = mask.reshape(1,mask.shape[0], mask.shape[1], 1)


    # clip to range [0, H/W] to not violate img boundaries
    x0 = np.clip(x0, zero, max_x)
    x1 = np.clip(x1, zero, max_x)
    y0 = np.clip(y0, zero, max_y)
    y1 = np.clip(y1, zero, max_y) 
    
    # get pixel value at corner coords
    Ia = input_fmap[0,y0,x0,:]
    Ib = input_fmap[0,y1,x0,:]
    Ic = input_fmap[0,y0,x1,:]
    Id = input_fmap[0,y1,x1,:]

    # recast as float for delta calculation
    x0 = np.float32(x0)
    x1 = np.float32(x1)
    y0 = np.float32(y0)
    y1 = np.float32(y1)

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)


    # add dimension for addition
    wa = np.expand_dims(wa,axis=3)
    wb = np.expand_dims(wb,axis=3)
    wc = np.expand_dims(wc,axis=3)
    wd = np.expand_dims(wd,axis=3)
    
    wa[np.where(wa<0)]=0
    wb[np.where(wb<0)]=0
    wc[np.where(wc<0)]=0
    wd[np.where(wd<0)]=0
    
    # compute output
    output_fmap = (wa*Ia + wb*Ib + wc*Ic + wd*Id)
    #output_fmap = Ib
    outM = output_fmap.copy()
    outM = outM * mask

    mask0 = 1*(mask==0)
    conVals = Ia * mask0
    output_fmap = conVals + outM

    output_fmap = Ia
    return output_fmap



def LinesIntersectionForLargestBox(trans_rpoints, rect_pts, theta):

    def line(p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
        return A, B, -C
    
    def intersection(L1, L2):
        D  = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return x,y
        else:
            return False

    rxymin = trans_rpoints.min(1)
    rxymax = trans_rpoints.max(1)

    widN = rxymax[0] - rxymin[0]
    higN = rxymax[1] - rxymin[1]
    
    rxyminOrig = rect_pts.min(1)
    rxymaxOrig = rect_pts.max(1)
    wid = rxymaxOrig[0] - rxyminOrig[0]
    hig = rxymaxOrig[1] - rxyminOrig[1]


# positive angle smaller width
    if wid<=hig and theta>=0:
        L1 = line([trans_rpoints[0][0], trans_rpoints[1][0]], [trans_rpoints[0][2], trans_rpoints[1][2]])
        L2 = line([trans_rpoints[0][1], trans_rpoints[1][1]], [trans_rpoints[0][3], trans_rpoints[1][3]])
        L3 = line([rxymin[0], rxymin[1]], [rxymax[0], rxymax[1]])
        #print('Condition 01 executed')
# positive angle greater width
    elif wid>hig and theta>=0:
        L1 = line([trans_rpoints[0][0], trans_rpoints[1][0]], [trans_rpoints[0][1], trans_rpoints[1][1]])
        L2 = line([trans_rpoints[0][2], trans_rpoints[1][2]], [trans_rpoints[0][3], trans_rpoints[1][3]])
        L3 = line([rxymax[0], rxymin[1]], [rxymin[0], rxymax[1]])
        #print('Condition 02 executed')

# negative angle greater width
    elif wid>hig and theta<0:# and (widOrig>higOrig):
        L1 = line([trans_rpoints[0][0], trans_rpoints[1][0]], [trans_rpoints[0][1], trans_rpoints[1][1]])
        L2 = line([trans_rpoints[0][2], trans_rpoints[1][2]], [trans_rpoints[0][3], trans_rpoints[1][3]])
        L3 = line([rxymin[0], rxymin[1]], [rxymax[0], rxymax[1]])
        #print('Condition 03 executed')
    
# negative angle smaller width
    elif (wid<=hig and theta<0): #or (widOrig<=higOrig):
        L1 = line([trans_rpoints[0][0], trans_rpoints[1][0]], [trans_rpoints[0][2], trans_rpoints[1][2]])
        L2 = line([trans_rpoints[0][1], trans_rpoints[1][1]], [trans_rpoints[0][3], trans_rpoints[1][3]])
        L3 = line([rxymax[0], rxymin[1]], [rxymin[0], rxymax[1]])
        #print('Condition 04 executed')
    

    if L1 and L3:
	intSec1 = intersection(L1, L3)
    if L2 and L3:
	intSec2 = intersection(L2, L3)

    if not intSec1 or not intSec2:
	intSec1 = [0, 0]
	intSec2 = [widN, higN]
    
    return intSec1, intSec2

def transformer_layer_fMap(input_fmap, angle, rpn_boxes, out_dims=None,  **kwargs):
    
  
    B = np.shape(input_fmap)[0]
    H = np.shape(input_fmap)[1]
    W = np.shape(input_fmap)[2]
    C = np.shape(input_fmap)[3]
    
   
    #cntr = np.asarray([(box[1]+box[3])/2, (box[0]+box[2])/2])
    cntr = np.asarray([H/2, W/2])
    #print('Original RPN :', box)

    T1 = [[1,0,0],[0,1,0],[-cntr[1],-cntr[0],1]]
    T2 = np.asarray([[np.cos(np.deg2rad(int(angle))), np.sin(np.deg2rad(int(angle))), 0],[-np.sin(np.deg2rad(int(angle))),np.cos(np.deg2rad(int(angle))),0],[0,0,1]])
    T3 = [[1,0,0],[0,1,0],[cntr[1],cntr[0],1]]
    

    T = np.dot(np.transpose(T3),np.dot(np.transpose(T2),np.transpose(T1)))
    #T = np.dot(np.transpose(T3),np.dot(np.transpose(T2),T1))

    
      
    corner_pts = [[0,W-1,0,W-1],[0,0,H-1,H-1],[1,1,1,1]]
    #print('corner_pts', corner_pts)

    trans_cpoints = np.dot(T[0:2,:],corner_pts)
    
    xymin = trans_cpoints.min(1)
    xymax = trans_cpoints.max(1)

    out_H = np.int32(xymax[1] - xymin[1] + 1)
    out_W = np.int32(xymax[0] - xymin[0] + 1)
    #print('out_W', out_W, out_H, W,H)

    out_fmap_size = [0,0,out_W, out_H]

    T4 = [[1,0,0],[0,1,0],[-xymin[0], -xymin[1],1]];
    #print T
    #print T4
   
    T_final = np.dot(np.transpose(T4),T);
    #print T_final
    

    tr_rotated_box_all = []

    for idx in range(0, len(rpn_boxes)):

	box = rpn_boxes[idx,1:]/16
	#print('box', box)

	rect_pts = [ [box[0],box[2], box[0], box[2]] ,[box[1],box[1],box[3],box[3]],[1,1,1,1]]

	trans_rpoints = np.dot(T_final[0:2,:],rect_pts)

	rxymin = trans_rpoints.min(1)
	rxymax = trans_rpoints.max(1)

	cropped_box = [np.int32(np.floor(rxymin[0])), np.int32(np.floor(rxymin[1])), np.int32(np.floor(rxymax[0])), np.int32(np.floor(rxymax[1]))]
	#cropped_box = [rxymin[0],rxymin[1],rxymax[0],rxymax[1]]

	# find coordinates for maximum area inscribed rectangle 
	intSec1, intSec2 = LinesIntersectionForLargestBox(trans_rpoints, np.array(rect_pts), angle)
	height_deltas = [intSec1[1]-cropped_box[1], cropped_box[3]-intSec2[1]]
        #print('height_deltas: ', height_deltas)

	tr_rotated_box = [rxymin[0], rxymin[1]+height_deltas[0], rxymax[0], rxymax[1]-height_deltas[1]]
	#print 'tr_cropped_box: ', tr_cropped_box
	tr_rotated_box = [ik * 16 for ik in tr_rotated_box]

	tr_rotated_box_all.append(tr_rotated_box)

   

    batch_grids = affine_grid_generator(out_H, out_W, T_final)
    
    x_s = batch_grids[:,0, :, :]
    y_s = batch_grids[:,1, :, :]

    out_fmap = bilinear_sampler(input_fmap, x_s, y_s)
    #print 'out_fmap ', out_fmap.shape
    #print 'input_fmap ', input_fmap.shape
    
    #tr_rotated_box_all = np.array(tr_rotated_box_all)
    return out_fmap, T_final, tr_rotated_box_all


def transformer_layer_fMapSep(input_fmap, orient_scores, rpn_boxes, out_dims=None,  **kwargs):
    
    theta = [0, 90, 135, 45, 157.5, 112.5, 67.5, 22.5]

    B = np.shape(input_fmap)[0]
    H = np.shape(input_fmap)[1]
    W = np.shape(input_fmap)[2]
    C = np.shape(input_fmap)[3]
    print('widHig', B, H, W, C)
    

    outMap = np.zeros((len(rpn_boxes), 72,72, input_fmap.shape[3]), dtype = float)
    #outMap = np.zeros((len(rpn_boxes), 50,50, input_fmap.shape[3]))


    tr_rotated_box_all = []
    transApplied = []
    #ang1 = np.array(np.argmax(orient_scores, axis = 1))
    #print('Im here :', 1*(ang1==0), 1*(ang1==1))
    #idx0 = np.where(np.logical_or((ang1 == 0)*1,(ang1 == 1)*1))[0]
    #print(idx0)

    for idx in range(0, len(rpn_boxes)):
	transCurrent = []

	angle = theta[np.argmax(orient_scores[idx, :], axis = 0)]

	if angle==0 or angle==90 :

	    #print ("input_fmap.shape",input_fmap.shape)
	    outMap[idx, 0:input_fmap.shape[1], 0:input_fmap.shape[2], 0:input_fmap.shape[3]] = input_fmap
	    #print rpn_boxes[idx,1:5], [idx]+[rpn_boxes[idx,1],rpn_boxes[idx,2], rpn_boxes[idx,3], rpn_boxes[idx,4]]
	    tr_rotated_box_all.append([idx]+[rpn_boxes[idx,1],rpn_boxes[idx,2], rpn_boxes[idx,3], rpn_boxes[idx,4]])

	    T11 = [[1,0,0],[0,1,0],[0,0,1]]
	    transCurrent.append(T11)
	    transCurrent.append(T11)
	    transApplied.append(transCurrent)

	    box = rpn_boxes[idx,1:5]/16		

	    sz = [box[3]-box[1], box[2]-box[0]]

	    cntr = np.asarray([(box[1]+box[3])/2, (box[0]+box[2])/2])

	    T1 = [[1,0,0],[0,1,0],[-cntr[1],-cntr[0],1]]
	    T2 = np.asarray([[np.cos(np.deg2rad(int(angle))), np.sin(np.deg2rad(int(angle))), 0],[-np.sin(np.deg2rad(int(angle))),np.cos(np.deg2rad(int(angle))),0],[0,0,1]])
	    T3 = [[1,0,0],[0,1,0],[cntr[1],cntr[0],1]]

	    T = np.dot(np.transpose(T3),np.dot(np.transpose(T2),np.transpose(T1)))

	    corner_pts = [[0,W-1,0,W-1],[0,0,H-1,H-1],[1,1,1,1]]

	    trans_cpoints = np.dot(T[0:2,:],corner_pts)

	    xymin = trans_cpoints.min(1)
	    xymax = trans_cpoints.max(1)

	    out_H = np.int32(xymax[1] - xymin[1] + 1)
	    out_W = np.int32(xymax[0] - xymin[0] + 1)

	    out_fmap_size = [0,0,out_W, out_H]

	    T4 = [[1,0,0],[0,1,0],[-xymin[0], -xymin[1],1]];

	    T_final = np.dot(np.transpose(T4),T);

	else:
		if angle>90:
		    angle = angle-180

		box = rpn_boxes[idx,1:5]/16		
		#print('angle', angle)


		sz = [box[3]-box[1], box[2]-box[0]]
	    
	   
	    	cntr = np.asarray([(box[1]+box[3])/2, (box[0]+box[2])/2])

	    #print('Original RPN :', box)

		T1 = [[1,0,0],[0,1,0],[-cntr[1],-cntr[0],1]]
		T2 = np.asarray([[np.cos(np.deg2rad(int(angle))), np.sin(np.deg2rad(int(angle))), 0],[-np.sin(np.deg2rad(int(angle))),np.cos(np.deg2rad(int(angle))),0],[0,0,1]])
		T3 = [[1,0,0],[0,1,0],[cntr[1],cntr[0],1]]


		T = np.dot(np.transpose(T3),np.dot(np.transpose(T2),np.transpose(T1)))
		#T = np.dot(np.transpose(T3),np.dot(np.transpose(T2),T1))

	    
	      
		corner_pts = [[0,W-1,0,W-1],[0,0,H-1,H-1],[1,1,1,1]]
		#print('corner_pts', corner_pts)

		trans_cpoints = np.dot(T[0:2,:],corner_pts)

		xymin = trans_cpoints.min(1)
		xymax = trans_cpoints.max(1)

		out_H = np.int32(xymax[1] - xymin[1] + 1)
		out_W = np.int32(xymax[0] - xymin[0] + 1)
		#print('out_W', out_W, out_H, W,H)

		out_fmap_size = [0,0,out_W, out_H]

		T4 = [[1,0,0],[0,1,0],[-xymin[0], -xymin[1],1]];
		#print T
		#print T4

		T_final = np.dot(np.transpose(T4),T);

		#print T_final

		rect_pts = [ [box[0],box[2], box[0], box[2]] ,[box[1],box[1],box[3],box[3]],[1,1,1,1]]

		trans_rpoints = np.dot(T_final[0:2,:],rect_pts)

		rxymin = trans_rpoints.min(1)
		rxymax = trans_rpoints.max(1)

		cropped_box = [np.int32(np.floor(rxymin[0])), np.int32(np.floor(rxymin[1])), np.int32(np.floor(rxymax[0])), np.int32(np.floor(rxymax[1]))]
		#print('cropped_box :', cropped_box)
		#print('trans_rpoints :', trans_rpoints)
	

		# find coordinates for maximum area inscribed rectangle 
		intSec1, intSec2 = LinesIntersectionForLargestBox(trans_rpoints, np.array(rect_pts), angle)
		height_deltas = [intSec1[1]-cropped_box[1], cropped_box[3]-intSec2[1]]
		#print('height_deltas: ', height_deltas)

		T11 = [[1,0,0],[0,1,0],[-rxymin[0],-rxymin[1],1]]
		T11 = np.transpose(T11)

		rect_pts1 = [trans_rpoints[0], trans_rpoints[1],[1,1,1,1]]
		trans_rpoints = np.dot(T11[0:2,:],rect_pts1)
		rxymin1 = trans_rpoints.min(1)
		rxymax1 = trans_rpoints.max(1)

		#print ('trans_rpoints00 : ', trans_rpoints)
		tr_rotated_box = [rxymin1[0], rxymin1[1]+height_deltas[0], rxymax1[0], rxymax1[1]-height_deltas[1]]
		#print ('box : ', box)
		#print ('tr_rotated_box : ', tr_rotated_box)

		tr_rotated_box = [ik * 16 for ik in tr_rotated_box]
		#ross = [[0]+ il for il in rotated_rpns]
		tr_rotated_box_all.append([idx]+tr_rotated_box)
	
		transCurrent.append(T_final)
		transCurrent.append(T11)
	
		batch_grids = affine_grid_generator(out_H, out_W, T_final)

		x_s = batch_grids[:,0, :, :]
		y_s = batch_grids[:,1, :, :]

		out_fmap = bilinear_sampler_Interpol(input_fmap.copy(), x_s, y_s)

	    	if cropped_box[0] < 0:

	       		cropped_box[2] = int(cropped_box[2] - cropped_box[0])
	       		cropped_box[0] = int(cropped_box[0] - cropped_box[0])

	    	if cropped_box[1] < 0:

	       		cropped_box[3] = int(cropped_box[3] - cropped_box[1])
	       		cropped_box[1] = int(cropped_box[1] - cropped_box[1])
	    
		f_map = out_fmap[:, cropped_box[1]:cropped_box[3] , cropped_box[0]:cropped_box[2] ,:]
		#print('output_fmap', (f_map[0,:,:,:]).sum())
		outMap[idx, 0:f_map.shape[1], 0:f_map.shape[2], 0:f_map.shape[3]] = f_map
		#print('output_fmap1', (outMap[idx,:,:,:]).sum())
	    
	    #tr_rotated_box_all = np.array(tr_rotated_box_all)
	    #print('featureMap size : ', outMap.shape)

		transApplied.append(transCurrent)

    return outMap, tr_rotated_box_all, transApplied, T_final





###### backward ######

#def transformer_layer_fMapSep_backward(input_fmap, orient_scores, rpn_boxes, out_dims=None,  **kwargs):
def transformer_layer_fMapSep_backward(input_grad, orient_scores, in_rpn_boxes, out_dims=None,  **kwargs):
    
    theta = [0, 90, 135, 45, 157.5, 112.5, 67.5, 22.5]

    B = np.shape(input_grad)[0]
    H = np.shape(input_grad)[1]
    W = np.shape(input_grad)[2]
    C = np.shape(input_grad)[3]
    print('widHig', B, H, W, C)
    

    outMap = np.zeros((len(in_rpn_boxes), 102,102, input_grad.shape[3]), dtype = float)
    #outMap = np.zeros((len(rpn_boxes), 50,50, input_fmap.shape[3]))


    tr_rotated_box_all = []
    transApplied = []
    #ang1 = np.array(np.argmax(orient_scores, axis = 1))
    #print('Im here :', 1*(ang1==0), 1*(ang1==1))
    #idx0 = np.where(np.logical_or((ang1 == 0)*1,(ang1 == 1)*1))[0]
    #print("len",len(in_rpn_boxes))

    for idx in range(0, len(in_rpn_boxes)):
	transCurrent = []

	angle = theta[np.argmax(orient_scores[idx, :], axis = 0)]
	#print ("angle", angle)



	if angle==0 or angle==90 :
	    #print ("input_grad.shape",input_grad.shape)
	    outMap[idx, 0:input_grad.shape[1], 0:input_grad.shape[2], 0:input_grad.shape[3]] = input_grad[idx, :, :, :]
	    #print rpn_boxes[idx,1:5], [idx]+[rpn_boxes[idx,1],rpn_boxes[idx,2], rpn_boxes[idx,3], rpn_boxes[idx,4]]
	    tr_rotated_box_all.append([idx]+[in_rpn_boxes[idx,1],in_rpn_boxes[idx,2], in_rpn_boxes[idx,3], in_rpn_boxes[idx,4]])

	    T11 = [[1,0,0],[0,1,0],[0,0,1]]
	    transCurrent.append(T11)
	    transCurrent.append(T11)
	    transApplied.append(transCurrent)



	    box = in_rpn_boxes[idx,1:5]/16		


	    sz = [box[3]-box[1], box[2]-box[0]]


	    cntr = np.asarray([(box[1]+box[3])/2, (box[0]+box[2])/2])


	    T1 = [[1,0,0],[0,1,0],[-cntr[1],-cntr[0],1]]
	    T2 = np.asarray([[np.cos(np.deg2rad(int(angle))), np.sin(np.deg2rad(int(angle))), 0],[-np.sin(np.deg2rad(int(angle))),np.cos(np.deg2rad(int(angle))),0],[0,0,1]])
	    T3 = [[1,0,0],[0,1,0],[cntr[1],cntr[0],1]]

	    T = np.dot(np.transpose(T3),np.dot(np.transpose(T2),np.transpose(T1)))

	    corner_pts = [[0,W-1,0,W-1],[0,0,H-1,H-1],[1,1,1,1]]

	    trans_cpoints = np.dot(T[0:2,:],corner_pts)

	    xymin = trans_cpoints.min(1)
	    xymax = trans_cpoints.max(1)

	    out_H = np.int32(xymax[1] - xymin[1] + 1)
	    out_W = np.int32(xymax[0] - xymin[0] + 1)

	    out_fmap_size = [0,0,out_W, out_H]

	    T4 = [[1,0,0],[0,1,0],[-xymin[0], -xymin[1],1]];

	    T_final = np.dot(np.transpose(T4),T);
	    T_final_inv = inv(T_final)


	else:
		if angle>90:
		    angle = angle-180

		box = in_rpn_boxes[idx,1:5]/16		
		#print('angle', angle)


		sz = [box[3]-box[1], box[2]-box[0]]
	    
	   
	    	cntr = np.asarray([(box[1]+box[3])/2, (box[0]+box[2])/2])

	    #print('Original RPN :', box)

		T1 = [[1,0,0],[0,1,0],[-cntr[1],-cntr[0],1]]
		T2 = np.asarray([[np.cos(np.deg2rad(int(angle))), np.sin(np.deg2rad(int(angle))), 0],[-np.sin(np.deg2rad(int(angle))),np.cos(np.deg2rad(int(angle))),0],[0,0,1]])
		T3 = [[1,0,0],[0,1,0],[cntr[1],cntr[0],1]]


		T = np.dot(np.transpose(T3),np.dot(np.transpose(T2),np.transpose(T1)))
		#T = np.dot(np.transpose(T3),np.dot(np.transpose(T2),T1))

		corner_pts = [[0,W-1,0,W-1],[0,0,H-1,H-1],[1,1,1,1]]
		#print('corner_pts', corner_pts)

		trans_cpoints = np.dot(T[0:2,:],corner_pts)

		xymin = trans_cpoints.min(1)
		xymax = trans_cpoints.max(1)

		out_H = np.int32(xymax[1] - xymin[1] + 1)
		out_W = np.int32(xymax[0] - xymin[0] + 1)
		#print('out_W', out_W, out_H, W,H)

		out_fmap_size = [0,0,out_W, out_H]

		T4 = [[1,0,0],[0,1,0],[-xymin[0], -xymin[1],1]];
		#print T
		#print T4

		T_final = np.dot(np.transpose(T4),T);
		T_final_inv = inv(T_final)

		#print T_final

		rect_pts = [ [box[0],box[2], box[0], box[2]] ,[box[1],box[1],box[3],box[3]],[1,1,1,1]]

		trans_rpoints = np.dot(T_final_inv[0:2,:],rect_pts)

		rxymin = trans_rpoints.min(1)
		rxymax = trans_rpoints.max(1)

		cropped_box = [np.int32(np.floor(rxymin[0])), np.int32(np.floor(rxymin[1])), np.int32(np.floor(rxymax[0])), np.int32(np.floor(rxymax[1]))]
		#print('cropped_box :', cropped_box)
		#print('trans_rpoints :', trans_rpoints)
	

		# find coordinates for maximum area inscribed rectangle 
		#intSec1, intSec2 = LinesIntersectionForLargestBox(trans_rpoints, np.array(rect_pts), angle)
		#height_deltas = [intSec1[1]-cropped_box[1], cropped_box[3]-intSec2[1]]
		#print('height_deltas: ', height_deltas)

		T11 = [[1,0,0],[0,1,0],[-rxymin[0],-rxymin[1],1]]
		T11 = np.transpose(T11)

		rect_pts1 = [trans_rpoints[0], trans_rpoints[1],[1,1,1,1]]
		trans_rpoints = np.dot(T11[0:2,:],rect_pts1)
		rxymin1 = trans_rpoints.min(1)
		rxymax1 = trans_rpoints.max(1)

		#print ('trans_rpoints00 : ', trans_rpoints)
		#tr_rotated_box = [rxymin1[0], rxymin1[1]+height_deltas[0], rxymax1[0], rxymax1[1]-height_deltas[1]]
		tr_rotated_box = [rxymin1[0], rxymin1[1], rxymax1[0], rxymax1[1]]
		#print ('box : ', box)
		#print ('tr_rotated_box : ', tr_rotated_box)


		tr_rotated_box = [ik * 16 for ik in tr_rotated_box]
		#ross = [[0]+ il for il in rotated_rpns]
		tr_rotated_box_all.append([idx]+tr_rotated_box)
	
		transCurrent.append(T_final_inv)
		transCurrent.append(T11)
	
		batch_grids = affine_grid_generator(out_H, out_W, T_final_inv)

		x_s = batch_grids[:,0, :, :]
		y_s = batch_grids[:,1, :, :]
		tup = np.reshape(input_grad[idx, :, :, :], (1,np.shape(input_grad)[1],np.shape(input_grad)[2],np.shape(input_grad)[3]))
		out_fmap = bilinear_sampler_Interpol(tup.copy(), x_s, y_s)
		#print ("tup.shape",tup.shape)
		#print (xyz)

	    	if cropped_box[0] < 0:

	       		cropped_box[2] = int(cropped_box[2] - cropped_box[0])
	       		cropped_box[0] = int(cropped_box[0] - cropped_box[0])

	    	if cropped_box[1] < 0:

	       		cropped_box[3] = int(cropped_box[3] - cropped_box[1])
	       		cropped_box[1] = int(cropped_box[1] - cropped_box[1])
	    
		f_map = out_fmap[:, cropped_box[1]:cropped_box[3] , cropped_box[0]:cropped_box[2] ,:]
		#print('output_fmap', (f_map[0,:,:,:]).sum())
		outMap[idx, 0:f_map.shape[1], 0:f_map.shape[2], 0:f_map.shape[3]] = f_map
		#print('output_fmap1', (outMap[idx,:,:,:]).sum())
	    
	    #tr_rotated_box_all = np.array(tr_rotated_box_all)
	    #print('featureMap size : ', outMap.shape)

		transApplied.append(transCurrent)

    return outMap, tr_rotated_box_all, transApplied
