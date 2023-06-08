#!/usr/bin/python

from __future__ import print_function, division

import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from skimage import io, color

def sparse_edit_optimization( palette, weights, target_colors,  palette_cons, verbose=True ):
    '''
    Given:
        palette: #palette -by-3 array (in global space)
        weights: k-by-#palette  array (k: number of constraints)
        target_color:  RGB colors as k-by-3 array
        palette_cons: a list of list of palette constraints. format: [ [palette color index,  selected change], ... ]
        #matrix: transformation matrix (parent->child) for optimizing sparsity of local palette
        verbose: use for displaying optimization details

    Returns:
        new_palette: An updated `palette` such that
            `weights` @ `palette` = `target_color`.
            The change to `palette` minimizes the L21 norm.
        check: a boolean indicating whether the optimizer successes or not.
    '''
    if weights is not None:
        assert palette.shape[1] == target_colors.shape[1] == 3
        assert weights.shape[1] == palette.shape[0] 
        assert target_colors.shape[0] == weights.shape[0]
    
    if verbose:
        print( "---input palette:\n", palette )
        if weights is not None:
            print( "---initial weights:", np.sum( weights, axis=1 ), '\n', weights )
            print( "---initial colors:\n", weights @ palette )
            print( "---target colors:\n", target_colors )
    
    ### objective function
    def l21( vec_delp ):
        delp = vec_delp.reshape( palette.shape )
        return sum( np.linalg.norm( delp, axis = 1 ) )     # optimize for local palette's sparsity
    
    ### setting up inequality constraint
    def color_constraints( vec_delp ):
        delp = vec_delp.reshape( palette.shape )
        rec_colors = weights @ (palette + delp)
        delta_colors = color.rgb2lab( rec_colors ) - color.rgb2lab( target_colors )
        return 2.29 * np.ones( delta_colors.shape[0] ) - np.linalg.norm( delta_colors, axis=1 )     # 2.29 > xxxxx
    
    ### setting up palette constraints
    def palette_constraints( vec_delp ):
        delp = vec_delp.reshape( palette.shape )
        rec_palette = palette + delp
        indices = [ cons[0] for cons in palette_cons ] 
        target_palettes = np.zeros( ( len( indices ), 3 ) )
        for i in range( len( indices ) ):
            target_palettes[i, :] = np.array( palette_cons[i][1] )
        return np.linalg.norm( rec_palette[ indices, : ] - target_palettes, axis = 1 )  # the norm of the change of selected palette colors should be zero
    
    ### setting up boundary condition and constraints
    if len( palette_cons ) != 0 and weights is not None:        # if there are both pixel and palette constraints
        cons = [{'type':'ineq', 'fun': color_constraints}, {'type':'eq', 'fun': palette_constraints}]
    elif len( palette_cons ) != 0 and weights is None:              # if there is only palette constraint
        cons = [{'type':'eq', 'fun': palette_constraints}]
    elif len( palette_cons ) == 0 and weights is not None:               # if there is only color constraint
        cons = [{'type':'ineq', 'fun': color_constraints}]
    else:   # if there are no constraints
        cons = []
        
    bounds = Bounds( - palette.flatten(), 1 - palette.flatten() )

    ### optimization
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    
    x0 = np.zeros( palette.shape[0] * palette.shape[1] )
    res = minimize( l21, x0, method='SLSQP', constraints=cons, options={'ftol': 1e-6, 'disp': False, 'maxiter': 300}, bounds=bounds )
    delp = ( res.x ).reshape( palette.shape )
    
    ### post-checking
    if verbose:
        np.set_printoptions(suppress=True)
        print('\n---difference:\n', np.linalg.norm( delp, axis=1 ) )
        print( '\n---new palette:\n', palette + delp )
        if  weights is not None:
            print( '\n---reconstructed colors:\n', weights @ (palette + delp) )
    
    ### check if we achieve color constraints
    if weights is not None:
        error = np.linalg.norm( color.rgb2lab( target_colors ) - color.rgb2lab( weights @ (palette + delp) ), axis=1 )
        check = True if ( error <= 2.32 ).all() else False  # use "2.32" to avoid numerical issues
    else:   # if there are only palette constraints, it's trivial optimization so always return True
        check = True
    
    return (palette + delp), check
    
if __name__ == '__main__':
    ### test OK!
    
    p = np.array([ [ 0., 0., 0. ], 
                            [ 0.313098, 0.35560563, 0.67021855 ], 
                            [ 1., 1., 1. ], 
                            [ 0.14887265, 0.261517, 0.12812837 ],  
                            [ 0.52481125, 0., 0. ],
                            [ 0.90326325, 0.84565842, 0.43033612 ] ])
                        
    #w = np.array([[0., 0.05227634, 0.30766621, 0., 0.26997928, 0.37007817], [0, 0.1, 0.1, 0.5, 0.3, 0]])
    w = np.array([[0., 0.05227634, 0.30766621, 0., 0.26997928, 0.37007817]])
    
    #target = np.array( [[0.67843137, 0.81960784, 0.41176471], [0.2, 0.3, 0.4]] )
    target = np.array( [[0.67843137, 0.81960784, 0.41176471]] )
    
    # test pixel constraint
    #opt_p, epsilon = sparse_edit_optimization( p, w, target, [0, 3] )
    #print( epsilon )
    
    # test palette constraint
    #opt_p, epsilon = sparse_edit_optimization( p, None, None, [ [3, [ 0.324, 0.59869, 1 ] ], [4, [ 0.324, 0.59869, 1 ] ], [5, [ 0.324, 0.59869, 1 ] ]] )