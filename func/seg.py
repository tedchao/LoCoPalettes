#!/usr/bin/env python3

import sys
import os
import cv2
import numpy as np

try:
    from .aux.weights import *
    from .aux.simplepalettes import *
except ImportError:
    from aux.weights import *
    from aux.simplepalettes import *
    

np.set_printoptions(suppress=True)

threshold = 0.0

class Palette_tree():
    def __init__( self, labels, tree, tracker, image, palette, weights=None, method='fea' ):
        self.labels = labels    # node masks \in [0, 1]
        self.parent2child = tree
        self.constraints_tracker = tracker
        self.activation = {}
        
        self.image = image      # need to update this at each iteration when user finish edits
        self.image_frozen = image
        self.global_palette = palette
        self.global_weights = weights
        self.rgbxy = None
        self.rgbfeaxy = None
        #self.pca_rgbfeaxy = None
        
        ### Palette, weights for sub-palette w.r.t. global palette
        self.node2palette = {}
        self.node2weight = {}
        
        self.method = method
    
    ############################################################################################################
    # PALETTE EXTRACTION BELOW
    ############################################################################################################
    
    def get_colors_from_node( self, node ):     ### tested.
        # Get colors from mask on specific node
        # the given mask on the node has values in [0,1]
        # where each value is the `alpha` of a specific color
        soft_mask = self.labels[node]
        #layer = self.image * soft_mask[:, :, np.newaxis]
        return self.image[ soft_mask > threshold ]
        #return layer[ soft_mask > 0. ]
    
    def compute_sub_palette( self, node ):      ### tested.
        # This could be computed by using [Wang et al. 2019]'s vertex refinement
        # extract a sub-palette for specific node
        colors = self.get_colors_from_node( node )
        og_hull = ConvexHull( colors )
        hvertices, hfaces = get_faces_vertices( og_hull )
        return simplified_convex_hull( 6, hvertices, hfaces, '', False, False ).vs
    
    def fill_sub_palette_in_tree( self, node ):     ### tested.
        # Fill all sub-palette in whole tree
        print('---Node:', node)
        if node == 0:
            self.node2palette[ node ] = self.global_palette
        else:
            self.node2palette[ node ] = np.unique( self.compute_sub_palette( node ), axis=0 )
        
        if node not in self.parent2child:   # if it's a leaf node
            return
        else:   # if it's also a parent node
            for child in self.parent2child[ node ]:
                self.fill_sub_palette_in_tree( child )
    
    ############################################################################################################
    # WEIGHTS COMPUTATION BELOW
    ############################################################################################################
    def compute_global_weight( self ):  # not used for now
        img_rgbxy = RGB_to_RGBXY( self.image )
        weights = np.array( RGBXY_weights( self.global_palette, img_rgbxy ) )
        self.global_weights = weights.reshape( (self.image.shape[0], self.image.shape[1], 6) )
    
    def compute_weights_from_node( self, node ):   ### tested
        colors = self.get_colors_from_node( node )
        palette = self.node2palette[ node ]
        rgbxy = self.rgbxy[ self.labels[ node ] > threshold ]   # get corresponding rgbxy
        #rgbfeaxy = self.rgbfeaxy[ self.labels[ node ] > threshold ]   # get corresponding rgbfeaxy
        
        # hard-code dimension. Never mind for now.
        if self.method == 'fea':
            return np.array( RGBXY_weights( palette, rgbxy, self.rgbfeaxy.reshape( -1, 133 ), ( self.image.shape[0], self.image.shape[1] ), self.labels[ node ]>threshold ) )
        if self.method == '18':
            return np.array( RGBXY_weights( palette, rgbxy ) )
    
    def fill_sub_weight_in_tree( self, node ):      ### tested.
        # Fill all sub-weight in whole tree
        print('---Node:', node)
        if node == 0:
            #self.compute_global_weight()
            self.node2weight[ node ] = self.global_weights.reshape( -1, 6 )
        else:
            self.node2weight[ node ] = self.compute_weights_from_node( node )
            
        if node not in self.parent2child:
            return
        else:
            for child in self.parent2child[ node ]:
                self.fill_sub_weight_in_tree( child )
    
    ############################################################################################################
    # FILL EVERYTHING NEEDED IN TREE
    ############################################################################################################
    def fill_activation( self ):
        for node in self.constraints_tracker:
            if node == 0:   # if root node, make it activated
                self.activation[ node ] = True
            else:   # only activate root node
                self.activation[ node ] = False
                
    def fill_all( self ):       ### tested.
        # Call the functions above and fill all necessary
        # informations in whole tree
        print( 'Compute local palettes...' )
        self.fill_sub_palette_in_tree(0)
        print( 'Compute local weights...' )
        self.fill_sub_weight_in_tree(0)
        self.fill_activation()
    
    ############################################################################################################
    # IDENTIFYING EDITED NODE BELOW
    ############################################################################################################
    
    def click_in_region( self, click, node ):       ### tested.
        # Check if a given click is within the node or not
        soft_mask = self.labels[ node ]
        return True if soft_mask[ click ] > 0.7 else False  # `0.7` is to enable to indicate a specific location corresponding to user's click
    
    ############################################################################################################
    # NEW FUNCTIONS
    ############################################################################################################
    def find_deepest_activated_node_contains_cons( self, cons ):
        candidates = []
        for node in self.constraints_tracker:
            if self.click_in_region( cons, node ) and self.activation[ node ]:
                candidates.append( node )
        return max( candidates )
    
    def find_deepest_node_contains_cons( self, cons ):
        candidates = []
        for node in self.constraints_tracker:
            if self.click_in_region( cons, node ):
                candidates.append( node )
        return max( candidates )
    
    def get_path_to_node( self, node, path=[] ):
        if node == 0:
            return [0] + path
        for parent, children in self.parent2child.items():
            if node in children:
                return self.get_path_to_node( parent, [node]+path )
            
    def check_not_contain_prev_level_cons( self, node ):
        # Use `constraint_tracker` to check if `node`'s region contains
        # previous level's contraints
        path_to_root = self.get_path_to_node( node )  
        path_to_root.pop( path_to_root.index(node) )    # exclude the node itself
        for n in path_to_root:
            for prev_con in self.constraints_tracker[ n ]:
                if self.click_in_region( prev_con, node ):
                    return False
        return True
    
    def find_cons_child_node_not_contain_prev_level_cons( self, cons, prev_node ):
        # Goal is to find one of children node of `node` that this children node does
        # not contain other constraints in previous level. (same level is fine.)
        for node in self.constraints_tracker:
            if node != prev_node and len( self.constraints_tracker ) != 0 and self.click_in_region( cons, node ) and self.check_not_contain_prev_level_cons( node ):
                return node
    
    ############################################################################################################
    # RECONSTRUCTION BELOW
    ############################################################################################################
    
    def paste_alpha_colors( self, background, local_colors, node ):
        '''
        Given: 
            background: background colors
            local_colors: pixel colors controlled by local palette
            node: a specific node location
        
        Return:
            composite: a composited image with soft boundaries
        '''
        soft_mask = self.labels[ node ]
        soft_mask = soft_mask[:, :, np.newaxis]
        alphas = np.repeat( soft_mask, 3, axis = 2 )
        composite = np.multiply( alphas, local_colors ) + np.multiply( 1-alphas, background )
        return composite
        
    def reconstruct_image_from_leaves( self, node, empty_image ):   # latest one
        # reconstruct image from leaf nodes with upward pass
        if node not in self.parent2child:
            soft_mask = self.labels[ node ]
            # get local weights
            local_weights = self.node2weight[ node ]
            # reconstruct colors in local region
            P_sub = self.node2palette[ node ]
            colors = local_weights @ P_sub
            # fill colors into a temporary image
            recon_image = np.zeros( self.image.shape )
            #print(colors.shape)
            recon_image[ soft_mask > threshold ] = colors
            # composite them with latest composited image
            recon_image = self.paste_alpha_colors( empty_image, recon_image, node )
            empty_image[ soft_mask > threshold ] = recon_image[ soft_mask > threshold ]
        else:
            for child in self.parent2child[ node ]:
                self.reconstruct_image_from_leaves( child, empty_image )
    
    ### Solve Eq. 4
    def solve_local_palette( self, parent, child, constrained = True ):
        '''
        Given:
            `parent`: a parent node
            `child`: a child node
        Return:
            `opt_local_palette`: an optimized local palette for `child` node
        '''
        import cvxpy as cp
        
        P_p, P_c = self.node2palette[ parent ], self.node2palette[ child ]
        
        empty_weights_in_image_size = np.zeros( ( self.labels[0].shape[0], self.labels[0].shape[1], self.node2weight[ 0 ].shape[1] ) )
        empty_weights_in_image_size[ self.labels[ parent ] > 0.0 ] = self.node2weight[ parent ]
        #W_p = self.node2weight[ parent ].reshape(  ( self.labels[0].shape[0], self.labels[0].shape[1], self.node2weight[ 0 ].shape[1] ) )
        W_c = self.node2weight[ child ]
        W_p_at_c = empty_weights_in_image_size[ self.labels[ child ] > 0.0 ]
        A = W_c.T @ W_c
        b = ( W_c.T @ W_p_at_c  ) @ P_p
        
        if not constrained:
            # solve it directly might be out-of-gamut
            opt_local_palette = np.linalg.solve( A, b )
            return opt_local_palette
        
        else:
            opt_local_palette = np.zeros( P_c.shape )
            for i in range( 3 ):
                x = cp.Variable( A.shape[0] )
                prob = cp.Problem( cp.Minimize( cp.quad_form( x, A ) - 2 * b[:, i].T @ x ),
                            [ np.zeros( A.shape[0] ) <= x, x <= np.ones( A.shape[0] ) ] )
                prob.solve()
                opt_local_palette[:, i] = x.value
            return opt_local_palette.clip( 0, 1 )
        
    def update_sub_palettes_under_a_node( self, node ):    ### tested.
            '''
            Given: 
                node: a given node
                palette: an updated palette
            Return:
                nothing to return. The function updates all children palettes from a given node.
            '''
            if node not in self.parent2child:
                return
            else:
                for child in self.parent2child[ node ]:
                    # recurse and solve for local palettes only if child node is not activated
                    if not self.activation[ child ]:
                        self.node2palette[ child ] = self.solve_local_palette( node, child )
                        self.update_sub_palettes_under_a_node( child )
                    
    #################################################################################
    ### Auxiliary functions for palette manipulation panel
    #################################################################################
    
    def find_current_labeling( self ):
        # find current labeling that allows users to manipulate palettes in each segment
        # the labeling is computed for further edge detection to visualize segmentation and corresponding palette
        labeling = np.zeros( (self.image.shape[0], self.image.shape[1]) )
        ordered_nodes = sorted( self.constraints_tracker )
        for node in ordered_nodes:
            if self.activation[ node ] and node != 0:
                labeling[ self.labels[ node ] > threshold ] = node
        return labeling.astype(np.uint8)

    def get_composited_image_and_palettes( self, image, labeling ):
        '''
        Given: 
            image: an original image
            labeling: an image with labels corresponding to current constraints
        
        Return:
            composite: a composited image with current palettes
        '''
        composite = (0.8 * 255. * np.ones( image.shape ) + 0.2 * 255. * image).round().clip(0, 255).astype(np.uint8)
        edges = cv2.Canny( labeling, 1, 2 )   
        
        # place segment edges
        composite[ edges > 0. ] = np.zeros( (1,3) ) 
        
        # iterate through each segment and place its corresponding local palettes
        for node in np.unique(labeling):
            x, y = np.where( labeling == node )
            center = ( int(np.mean(x)), int(np.mean(y)) )
            
            # farthest point and its closest distance to boundary
            fpt, radius = self.get_farthest_point( (x,y) ) 
            
            # locate local palette
            palette = np.ascontiguousarray(palette2swatch( self.node2palette[ node ] )*255.).round().clip(0, 255).astype(np.uint8)
            palette_img = np.ascontiguousarray( cv2.resize(palette, dsize=(int(radius), int(radius/5)), interpolation=cv2.INTER_NEAREST) )
            
            # place palette towards center of each segment
            w_palette, h_palette = palette_img.shape[0], palette_img.shape[1]
            composite[ fpt[0]-w_palette // 2: fpt[0]-w_palette // 2+w_palette, fpt[1]-h_palette // 2: fpt[1]-h_palette // 2+h_palette ] = palette_img
        return composite
    
    def get_farthest_point( self, loc ):
        '''
        Given:
            loc: a tuple. First element is the x locations of a specific labeling. Second is the y's.

        Return:
             farthest point inside this specific segment using distance transform
        '''
        aux = np.zeros( (self.image.shape[0], self.image.shape[1]) )
        aux[ loc[0], loc[1] ] = 1
        
        # helping augmented 2D array
        aug = np.zeros( (self.image.shape[0]+2, self.image.shape[1]+2) )
        aug[1:-1, 1:-1] = aux
        
        # distance transform to find farthest point
        dist = cv2.distanceTransform( aug.astype(np.uint8), cv2.DIST_L2, 5 )
        
        ind = np.unravel_index( dist.argmax(), dist.shape )
        return ( ind[0]-1, ind[1]-1 ), np.max(dist)    
    
    def find_node_from_current_constraints_on_click( self, click ):
            '''
            **Find node from deepest level that contains the click and has other constraint.
            
            Given:
                click: a user's click on control panel
            
            Return:
                node: a node location specific to user's click
            '''
            # if some constraints are placed, we traverse the tree from deepest level and find node location
            ordered_nodes = sorted( self.constraints_tracker )
            ordered_nodes.reverse()     # reverse it to traverse from deepest level
            for node in ordered_nodes:
                if self.activation[ node ] and self.click_in_region( click, node ):
                    return node
            return 0    # if no further lower level constraint, then it's modifying the root palette
    
## Some auxiliaries
def copy_tree( ptree ):
    import copy
    new_tree = Palette_tree( copy.deepcopy( ptree.labels ), copy.deepcopy( ptree.parent2child ), copy.deepcopy( ptree.constraints_tracker ),
                                        copy.deepcopy( ptree.image ), copy.deepcopy( ptree.global_palette ), copy.deepcopy( ptree.global_weights ) )
    new_tree.node2palette = copy.deepcopy( ptree.node2palette )
    new_tree.node2weight = copy.deepcopy( ptree.node2weight )
    new_tree.activation = copy.deepcopy( ptree.activation )
    return new_tree

def main():
    import scipy.io as sio
    import argparse
    parser = argparse.ArgumentParser( description = 'Quadtree testing.' )
    parser.add_argument( 'input', help = 'The path to the input image.' )
    parser.add_argument( 'features', help = 'The path to the input features.' )
    parser.add_argument( 'folder', help = 'The path to the folder for soft segments.' )
    parser.add_argument( '--o', '--option', help = 'Option for saving or loading soft segments.' )
    parser.add_argument( '--m', '--method', help = 'Option for saving or loading soft segments.' )
    args = parser.parse_args()
    
    image = cv2.cvtColor( cv2.imread(args.input), cv2.COLOR_BGR2RGB ) / 255.
    print( 'image size:', image.shape )
    
    features = sio.loadmat(args.features)['embedmap']
    print( 'feature size:', features.shape )
    
    
    import seg2tree
    quadtree = seg2tree.get_user_tree( args.folder )
    
    ### initialize palette tree
    import json
    
    # if user wants to save palette tree
    if args.o == 's':
        print( 'Compute global palette...' )
        # extract palette
        og_hull = ConvexHull( image.reshape(-1,3) )
        hvertices, hfaces = get_faces_vertices( og_hull )
        palette = simplified_convex_hull( 6, hvertices, hfaces, '', True, False ).vs
        
        print( 'Compute global weights...' )
        # extract weights
        img_rgbxy = RGB_to_RGBXY( image )
        
        if args.m == 'fea':
            weights = np.array( RGBXY_weights( palette, img_rgbxy, concat_RGBFEAXY( image, features ).reshape( -1, features.shape[2]+5 ) ) )
        if args.m == '18':
            weights = np.array( RGBXY_weights( palette, img_rgbxy ) )
        
        # reconstruct image
        global_weights = weights.reshape( (image.shape[0], image.shape[1], 6) )
        image = ( weights @ palette ).reshape( image.shape )
        
        # project pixels onto leaf palette and compute global weights in terms of new image
        ptree = Palette_tree( quadtree.labels, quadtree.parent2child, quadtree.constraints_tracker, image, palette, global_weights, method= args.m )
        
        # save rgbxy and rgbfeaxy
        ptree.rgbxy = img_rgbxy.reshape( ( image.shape[0], image.shape[1], 5 ) )
        ptree.rgbfeaxy = concat_RGBFEAXY( image, features )
        
        ptree.fill_all()
        save_for_test( ptree )
        print( 'Palette tree saved.' )
        
        #with open( 'weights-' + args.m + '.npy', 'wb' ) as f:
            #np.save( f, weights )
    
    # if user has its own palette tree already
    if args.o == 'l':
        # initialization
        ptree = Palette_tree( quadtree.labels, quadtree.parent2child, quadtree.constraints_tracker, image, None )
        load_for_test( ptree )
        print( "Palette tree loaded." )
        
        # reconstruction for testing
        recon = np.copy( image )
        ptree.reconstruct_image_from_leaves( 0, recon )
        cv2.imwrite( 'nesi-recon.png', cv2.cvtColor( ( 255.* recon ).astype( np.uint8 ), cv2.COLOR_RGB2BGR ) )
        
        # edits for testing
        #color = np.array( [ 0, 0, 255 ] ) / 255.     # color
        #ptree.node2palette[ 0 ][ 0 ] = color
        recon_global = ( ptree.node2weight[0] @ ptree.node2palette[0] ).reshape( image.shape )
        cv2.imwrite( 'nesi-edit-global-2.png', cv2.cvtColor( ( 255.* recon_global ).astype( np.uint8 ), cv2.COLOR_RGB2BGR ) )
        
        '''
        ptree.update_sub_palettes_under_a_node( 0 )
        recon = np.copy( image )
        ptree.reconstruct_image_from_leaves( 0, recon )
        cv2.imwrite( 'nesi-edit-prop-2.png', cv2.cvtColor( ( 255.* recon ).astype( np.uint8 ), cv2.COLOR_RGB2BGR ) )
        '''
        
        print( '------ basic reconstruction measurement: ' )
        print( 'global: ', np.linalg.norm( recon_global - image ) )
        print( 'local: ', np.linalg.norm( recon - image ) )
        
        print( '------ reconstruction measurement: ')
        #print( 'global:', np.linalg.norm(recon_global-image))
        #print( 'local:', np.linalg.norm(recon-image))
        diff = np.sqrt( ( ( recon - recon_global )**2 ).sum( axis = -1 ) )
        print( 'global versus local (min/avg/max):', diff.min(), np.average( diff ), diff.max() )
        print( 'global versus local (percentiles: 0/5/25/50/75/95/99/100):', *( np.percentile( diff, p ) for p in ( 0, 5, 25, 50, 75, 95, 99, 100 ) ) )
        #print( 'global versus local:', np.linalg.norm(recon-recon_global))
        
        ######
        print( '------ sparsity measurement: ')
        global_weights = ptree.node2weight[0]
        center_weights = (1/6) * np.ones( 6 )
        print( '(Aksoy) global sparsity:', np.sum( global_weights ) / np.sum( global_weights ** 2 ) - 1 )
        print( '(Tan) global sparsity:', ( 1 / ( global_weights.shape[0] * global_weights.shape[1] ) ) * np.sum( - ( 1 - global_weights ) **2 ) )
        print( '------------------' )
        
        '''
        for node in ptree.activation:
            if node != 0:
                print( 'Node: ', node )
                print( '(Aksoy) local sparsity:', np.sum( ptree.node2weight[node] ) / np.sum( ptree.node2weight[node] ** 2 ) - 1 )
                print( '(Tan) local sparsity:', ( 1 / ( ptree.node2weight[node].shape[0] * ptree.node2weight[node].shape[1] ) ) * np.sum( - ( 1 - ptree.node2weight[node] ) **2 ) )
        '''
        
        
def save_for_test( tree ):
    import json
    with open( 'tree.json', 'w' ) as f:
        json.dump( {node: children for node, children in tree.parent2child.items()}, f )
        
    with open( 'palette.json', 'w' ) as f:
        json.dump( {node: palette.tolist() for node, palette in tree.node2palette.items()}, f )

    with open( 'weights.json', 'w' ) as f:
        json.dump( {node: weight.tolist() for node, weight in tree.node2weight.items()}, f )
        
    with open( 'mask.json', 'w' ) as f:
        json.dump( {node: mask.tolist() for node, mask in tree.labels.items()}, f )
    
    with open( 'tracker.json', 'w' ) as f:
        json.dump( {node: children for node, children in tree.constraints_tracker.items()}, f )
        
    with open( 'activation.json', 'w' ) as f:
        json.dump( {node: children for node, children in tree.activation.items()}, f )
    
def load_for_test( ptree ):
    import json
    from ast import literal_eval
    
    with open( 'tree.json' ) as f:
        tree = {int(node): children for node, children in json.load(f).items() }
        
    with open( 'palette.json' ) as f:
        palettes = {int(node): np.array(palette) for node, palette in json.load(f).items() }
    
    with open( 'weights.json' ) as f:
        weights = {int(node): np.array(weight) for node, weight in json.load(f).items() }
        
    with open( 'mask.json' ) as f:
        masks = {int(node): np.array(labels) for node, labels in json.load(f).items() }
    
    with open( 'tracker.json' ) as f:
        tracker = {int(node): labels for node, labels in json.load(f).items() }
        
    with open( 'activation.json' ) as f:
        activation = {int(node): labels for node, labels in json.load(f).items() }
    
    ptree.parent2child = tree
    ptree.node2palette = palettes
    ptree.node2weight = weights
    ptree.labels = masks
    ptree.constraints_tracker = tracker
    ptree.activation = activation

if __name__ == '__main__':
    main()
    