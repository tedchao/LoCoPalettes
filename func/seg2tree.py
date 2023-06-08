#!/usr/bin/env python3

import cv2
import numpy as np
import os

# Class for User's Tree
class UTree():
	def __init__( self, labels, p2c, cons_tracker ):
		self.labels = labels
		self.parent2child = p2c
		self.constraints_tracker = cons_tracker

def get_labels_and_tree( dir ):
	# I/O image segments
	segs = []
	for filename in os.listdir(dir):
		if filename.endswith(".png") and filename != '0.png': 
			segs.append( filename )
	
	# initialization
	root = 1.0 - cv2.imread( dir + '/' + '0.png', 0 )
	parent2child = {0:[]}
	labels = {0: root}
	constraints_tracker = {}
	
	# construct tree structure and corresponding masks
	for s in segs:
		parent, child = s[: -4].split( '-' )
		
		# push mask info to `labels`
		mask = np.ones( root.shape )
		img = 1 - cv2.imread( dir + '/' + s, 0 ) / 255.
		#mask[ img==255 ] = 0 # open up this comment if you are not using grayscale
		#labels[ int(s[2]) ] = mask
		labels[ int( child ) ] = img
		
		if int( parent ) in parent2child:
			parent2child[ int( parent ) ].append( int( child ) )
		else:
			parent2child[ int( parent ) ] = [ int( child ) ]
		
	# sort children for simplicity
	for node in parent2child:
		parent2child[ node ].sort()
	
	# initialize constraint tracker
	for node in labels:
		constraints_tracker[ node ] = []
	
	return labels, parent2child, constraints_tracker

def get_user_tree( folder ):
	labels, p2c, cons_tracker = get_labels_and_tree( folder )
	return UTree( labels, p2c, cons_tracker )

#get_user_tree()
