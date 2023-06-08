#!/usr/bin/env python3
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import io

import torch
import torchvision.transforms as T
import numpy as np
import cv2
from cv2.ximgproc import guidedFilter
import math

import panopticapi
from panopticapi.utils import id2rgb, rgb2id

import warnings
warnings.filterwarnings("ignore")
torch.set_grad_enabled(False);

# standard PyTorch mean-std input image normalization
transform = T.Compose([
	T.Resize(800),
	T.ToTensor(),
	T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_hierarchy( model, postprocessor, image, folder_path, save_img=True ):
	'''
	Given: 
		`model`: DETR model
		`postprocessor`: post processor for getting labels from segmentation output
		`image`: PIL image
		`folder_path`: folder path for storing hierarchy
	Return:
		Nothing to return
	'''
	
	# mean-std normalize the input image (batch-size: 1)
	trans_image = transform( image ).unsqueeze(0)
	out = model( trans_image )
	
	# compute the scores, excluding the "no-object" class (the last one)
	scores = out["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]
	
	# threshold the confidence
	keep = scores > 0.85
	
	(h, w, c) = np.array( image ).shape
	
	# obtain prediction result
	result = postprocessor(out, torch.as_tensor((h,w)).unsqueeze(0))[0]
	
	# build preliminary hierarchy
	segments_info = result["segments_info"]
	uni_category, counts = np.unique( [seg["category_id"] for seg in segments_info], return_counts=True )
	
	# build tree (classes -> instances)
	max_id = len( uni_category )+1
	tree = { "0": list( range( 1, max_id ) ) }
	for i in range( len( counts ) ):
		if counts[i] > 1:
			tree[ str( tree["0"][i] ) ] = list( range( max_id, max_id+counts[i] ) )
			max_id += counts[i]
	
	# The segmentation is stored in a special-format png
	panoptic_seg = Image.open(io.BytesIO(result['png_string']))
	panoptic_seg = np.array(panoptic_seg, dtype=np.uint8).copy()
	# We retrieve the ids corresponding to each mask
	panoptic_seg_id = rgb2id(panoptic_seg)
	
	# build dictionary for category to masks 
	cat_2_masks = { cat: [] for cat in uni_category}
	for s in segments_info:
		seg = np.zeros( (h,w) )
		seg[panoptic_seg_id == s["id"]] = 1.0
		cat_2_masks[ s["category_id"] ].append( seg )
		
	# fill up masks
	masks = {}
	for node, children in tree.items():
		if node == "0":
			masks[ node ] = np.ones( (h,w) )
			# if it's root node, sum all the masks in its children
			for i in range( len( children ) ):
				category = uni_category[ children[i]-1 ]
				masks[ str( children[i] ) ] = sum( cat_2_masks[category] )
		else:	# otherwise, store it one-by-one
			category = uni_category[ int(node)-1 ]
			for i in range( len( children ) ):
				masks[ str( children[i] ) ] = cat_2_masks[category][i]
	
	if save_img:
		image = ( 255. * np.array( image ) ).astype( np.uint8 )
		
		# store the hierarchy
		for node, children in tree.items():
			if node == "0":
				cv2.imwrite( folder_path + node + '.png', ( 255.*( 1-masks[ node ] ) ).astype( np.uint8 ) )
			for child in children:
				# apply dilation to make segments overlapped
				
				
				kernel = np.ones( (5, 5), 'uint8' )
				seg = cv2.dilate( masks[ str( child ) ], kernel, iterations=1 )
				seg = ( 255. * ( 1-seg ) ).astype( np.uint8 )
				seg = guidedFilter( image,  seg,  5, 1e-6 )
				cv2.imwrite( folder_path + node + '-' + str( child ) + '.png', seg )
				
				#cv2.imwrite( folder_path + node + '-' + str( child ) + '.png', ( 255.*( 1-masks[ str( child ) ] ) ).astype( np.uint8 ) )
				
def main():
	import os
	import argparse
	parser = argparse.ArgumentParser( description = 'Panoptic segmentation.' )
	parser.add_argument( 'input', help = 'The path to the input image.' )
	args = parser.parse_args()
	
	# load image and features
	image = Image.open( args.input )
	
	# folder path for storing hierarchy
	if '/' in args.input:
		folder_path = 'sss_' + args.input.split('/')[1][:-4] + '/'
	else:
		folder_path = 'sss_' + args.input[:-4] + '/'
	
	if not os.path.exists( folder_path ):
		# Create a new directory because it does not exist 
		os.makedirs( folder_path )
		print("The new directory " + folder_path + " is created!")
	
	# load DETR model
	model, postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True, return_postprocessor=True, num_classes=250)
	model.eval()
	
	predict_hierarchy( model, postprocessor, image, folder_path )

if __name__ == '__main__':
	main()