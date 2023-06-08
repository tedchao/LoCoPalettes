import os

from .aux.simplepalettes import *
from .aux.simplify_convexhull import *
from .aux.weights import *

from .seg import *
from .optimizer import *

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull, Delaunay

from PIL import Image
import cv2

from scipy.optimize import minimize
from scipy.optimize import Bounds

from skimage import morphology
from skimage import io, color

############################################################################################################################

### Convert palette to palette-image compatible to be pasted on GUI
def palette_to_palette_img( palette ):
    palette_img = np.ascontiguousarray(palette2swatch( palette )*255.).round().clip(0, 255).astype(np.uint8)
    return palette_img

def image_2_image_panel( image ):
    return np.ascontiguousarray(image*255.).round().clip(0, 255).astype(np.uint8)

### simple image reconstruction
def get_recolor_img( palette, weights, img_shape ):
    return ( weights @ palette ).reshape( img_shape )
############################################################################################################################

### Palette extraction
def get_palette( palette_size, img_arr, img_path ):
    print( "Extracting palette..." )
    og_hull = ConvexHull( img_arr.reshape( ( -1, 3 ) ) )
    hvertices, hfaces = get_faces_vertices( og_hull )
    palette = simplified_convex_hull( palette_size, hvertices, hfaces, img_path ).vs
    print( "Palette extracted." )
    return palette

### Weights computation
def get_weights( palette, img_arr, features ):
    print( "Computing weights..." )
    rgbxy = RGB_to_RGBXY( img_arr )
    rgbfeaxy = concat_RGBFEAXY( img_arr, features ).reshape( -1, features.shape[2]+5 ) 
    del_coord = RGBXY_weights( palette, rgbxy, rgbfeaxy )      # ours
    #del_coord = RGBXY_weights( palette, rgbxy )         #[Tan et al. 2018]
    print( "Weights computed." )
    return del_coord


### Optimizer of palette hierarchy
def pixel_constraints_to_array( pixel_constraints, trackers ):
    # `pixel_constraints`: format: [(position, original weight, [original color, target color]), ...]
    # `trackers`: format: [position1, position2, ...]
    # return: np array weigths, np array target colors
    assert len( pixel_constraints ) != 0
    weights = np.zeros( (len(trackers), pixel_constraints[0][1].shape[0]) )
    target_colors = np.zeros( (len(trackers), 3) )
    
    # order matters, start from looping `trackers`
    for i in range( len(trackers) ):
        ind = [x[0] for x in pixel_constraints].index( trackers[i] )
        # stacks up constraints corresponding to that node
        weights[i, :] = pixel_constraints[ind][1]
        target_colors[i, :] = pixel_constraints[ind][2][1]
    return weights, target_colors


def pixel_constraints_to_array_local( pixel_constraints, cptree, node ):
    # `pixel_constraints`: format: [(position, [original color, target color], [localize_indicator]), ...]
    # return: # cons- by- 6 weights; # cons- by- 3 target colors
    node_constraints = cptree.constraints_tracker[ node ]
    num_cons = len( node_constraints )
    region_mask = cptree.labels[ node ]
    region_weight = np.zeros( ( region_mask.shape[0], region_mask.shape[1], cptree.node2weight[ node ].shape[1] ) )
    region_weight[ region_mask > 0. ] = cptree.node2weight[ node ]
    
    weights = np.zeros( ( num_cons, region_weight.shape[2] ) )
    target_colors = np.zeros( ( num_cons, 3 ) )
    for i in range( num_cons ):
        ind = [ x[0] for x in pixel_constraints ].index( node_constraints[i] )
        # stacks up constraints corresponding to that node
        pos = node_constraints[i]           # use small window to compute average weights for the given constraint
        #weights[i, :] = region_weight[ node_constraints[i] ]   
        window_local_weights = region_weight[ pos[0] - 1: pos[0] + 1, pos[1] - 1: pos[1] + 1, : ]
        weights[i, :] = np.mean( window_local_weights.reshape( -1, region_weight.shape[2] ), axis = 0 )
        target_colors[i, :] = pixel_constraints[ind][1][1]
    return weights, target_colors
    

def optimize_hierarchy( ptree, pixel_constraints, palette_constraints ):
    # `palette_constraints`: {0: [  (palette color index, location, [ selected change ]) ], 1: [  (palette color index, location, [ selected change ]) ], ...}
    
    # loop through nodes that contain constraints
    for node in ptree.constraints_tracker:
        if len( ptree.constraints_tracker[ node ] ) != 0:
            # get weights and target colors
            weights, target_colors = pixel_constraints_to_array_local( pixel_constraints, ptree, node )
            palette = ptree.node2palette[ node ]
            
            # get palette constraints
            palette_cons = [ [ pcons[0], pcons[2][0] / 255. ] for pcons in palette_constraints[ node ] ]
            
            print( '------------------------------Optimizing at node------------------------------: ', node )
            
            # apply optimizer
            opt_palette, check = sparse_edit_optimization( palette, weights, target_colors, palette_cons )
            
            if check:   # if optimization ends successfully, we update palette hierarchy
                ptree.node2palette[ node ] = opt_palette
                ptree.update_sub_palettes_under_a_node( node )  ### Solve Eq. 4
                # if optimization is good to go, activate the corresponding node so next constraint will be at least added to this node (if its mask contains the next constraint)
                ptree.activation[ node ] = True
            else:       # otherwise, stops further optimization and make arragements for current constraints on the hierarchy
                return node
        # if there is no pixel constraints but there are some palette constraints
        else:
            if len( palette_constraints[ node ] ) != 0:
                print( '------------------------------Optimizing palette at node------------------------------: ', node )
                
                # get palette constraints
                palette_cons = [ [ pcons[0], pcons[2][0] / 255. ] for pcons in palette_constraints[ node ] ]
                palette = ptree.node2palette[ node ]
                
                # apply optimizer
                opt_palette, check = sparse_edit_optimization( palette, None, None, palette_cons )
                if check:
                    ptree.node2palette[ node ] = opt_palette
                    ptree.update_sub_palettes_under_a_node( node )
    return -1

### Modify palette hierarchy
def modify_hierarchy( ptree, node ):
    # time to change the node positino of this specific constraints
    cons = ptree.constraints_tracker[ node ].pop()
    
    # find child node of current node
    # AND this child node does not contain any other constraint that is above child node's level
    node_to_opt = ptree.find_cons_child_node_not_contain_prev_level_cons( cons, node )
    
    # push this constraint to its children node for further optimization
    ptree.constraints_tracker[ node_to_opt ].append( cons )