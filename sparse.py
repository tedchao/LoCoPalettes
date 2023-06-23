import sys
import os

import numpy as np
import cv2
from PIL import Image

from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
from qtwidgets import Toggle

from func.utils import *
from func.seg import *

# In command line: "pip install opencv-python-headless" to avoid qt complaining two set of binaries

### This sometimes happens: 'qt.gui.icc: fromIccProfile: failed minimal tag size sanity'

## A workaround for a bug in Qt with Big Sur. Later Qt versions don't need this workaround.
## I don't know which version.
import platform
if platform.system() == 'Darwin' and platform.mac_ver()[0] >= '10.16':
    os.environ['QT_MAC_WANTS_LAYER'] = '1'

class Edits():
    def __init__( self ):
        ## all kinds of images
        self.imagePath = ""   # input image path
        self.image = None   # save image in cv2 format
        self.image_og = None    # original numpy image 
        self.image_frozen = None    # frozen copy of numpy image. used for linear color clipping.
        
        self.composite = None    # image for segmentation composited with palettes in palette tree
        self.recon_img = None
        
        ## palettes, weights and features
        self.palette = None # save palette in np format
        self.palette_np_og = None
        self.per_pixel_weights = None
        self.features = None
        
        ## helping variables for stopping optimizer to repeatly optimize
        self.prev_optimize_color = np.array([-1, -1, -1])
        self.prev_optimize_palette_color = np.array([-1, -1, -1])
        
        ## keep track of what users are doing now
        self.palette_click = False
        self.image_click = False
        self.sparse_edit_indicator = False
        
        ## all about palette tree
        self.ptree_baseline = None   # palette tree
        self.ptree_process = None  # palette tree that is currently under optimization
        self.ptree_reset = None    # frozen palette tree, use for displaying original local palettes
        self.node_local = 0
        
        ## holding constraint colors, positions, and weights
        self.pixel_constraints = []     # format: [(position, original weight, [original color, target color]) ]. Note: order matters! So dictionary is not suitable.
        self.palette_constraints = {}   # format: {0: [  (palette color index, location, [ selected change ]) ], 1: [  (palette color index, location, [ selected change ]) ], ...}
        
        self.user_colors = []
        self.user_pos = []
        self.user_weights = []
        self.ind_pixel = -1
        self.ind_palette = -1
    
    ### global palette computation
    def extract_palette( self, palette_size ):
        self.palette = get_palette( palette_size, self.image, self.imagePath )
        self.palette_np_og = self.palette
        self.frozen_palette = self.palette
    
    ### global weights computation
    def extract_weights( self ):
        self.per_pixel_weights = get_weights( self.palette, self.image, self.features )
        self.image = get_recolor_img( self.palette, self.per_pixel_weights, self.image.shape )
    
    ### optimizer takes a palette tree and uses existing pixel/palette constraints under optimization
    def sparse_edit_handler( self, ptree ):
        cptree = copy_tree( ptree )     # copy the baseline palette hierarchy every time for optimization
        split_indicator = False
        
        for cons in self.pixel_constraints:
            # push the constraint to deepest activated node that contains the constraint
            if not cons[2][0]:  
                #### TODO: modify it to just down to one level lower
                ### Ted: I still think it's true to push to deepest activated node
                ### reason: if we have node relationship p0 ->p1 -> p2
                ### cases: (1) if p1 is activated p2 is not: this will push to p1. no problem.
                ###            (2) if p2 is activated p1 is not: this will push to p2. no problem.
                ###            (3) if both p1 and p2 are activated: if the click is on p1 but not p2, this will push to p1. if the click is on p2 (within p1), we should push this click to p2. --> deepest
                node = cptree.find_deepest_activated_node_contains_cons( cons[0] )  
            else:   # if it's localized, then pushed it to the deepest node
                node = cptree.find_deepest_node_contains_cons( cons[0] )
            cptree.constraints_tracker[ node ].append( cons[0] )
            
            # start optimizing the whole hierarchy
            while True:
                ### Solve Eq. 1
                wrong_node = optimize_hierarchy( cptree, self.pixel_constraints, self.palette_constraints )
                if wrong_node == -1:   # if hierarchy is optimized with no errors (indicator: node as -1), then break
                    # Note that palette propagation can be implemented inside `optimize_hierarchy` to better benchmarking the local palettes
                    # (if further constraints are added)
                    break
                else:   # otherwise, start finding new node for this constraint until it finds new location that makes hierarchy error-free
                    split_indicator = True
                    modify_hierarchy( cptree, wrong_node )
        
        # if there's no pixel constraints, solve palette constraints if there are some
        if len( self.pixel_constraints ) == 0:
            wrong_node = optimize_hierarchy( cptree, self.pixel_constraints, self.palette_constraints )
            
        return cptree, split_indicator