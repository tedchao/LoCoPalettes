#!/usr/bin/env python3

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.sparse import coo_matrix
from scipy.optimize import nnls

from .simplify_convexhull import *

from PIL import Image
import cv2
import matplotlib.pyplot as plt

import pyximport
pyximport.install(reload_support=True)
from .GteDistPointTriangle import *
import warnings
warnings.filterwarnings("ignore")

def RGB_to_RGBXY( img ):
    '''
    Given: np array of images (shape: width, hegith, 3)
    
    Return: An image with RGBXY. (shape: width x hegith, 5)
    '''
    width, height = img.shape[0], img.shape[1]
    img_rgbxy = np.zeros( ( width, height, 5 ) )
    
    for i in range( width ):
        for j in range( height ):
            img_rgbxy[i, j, :3] = img[i, j]
            img_rgbxy[i, j, 3:5] = (1/max(width, height)) * np.array( [i, j] )
    img_rgbxy = img_rgbxy.reshape( (-1, 5) )
    return img_rgbxy

def concat_RGBFEAXY( image, features ):
    (h, w, c1) = image.shape
    (_, _, c2) = features.shape
    x, y = np.meshgrid( np.arange( h ), np.arange( w ) )   # position xy
    rgbfeaxy = np.zeros( (h, w, c1+c2+2) )
    rgbfeaxy[:, :, :c1] = image
    rgbfeaxy[:, :, c1:c1+c2] = features
    rgbfeaxy[:, :, c1+c2] = (1/max(h, w)) * x.T
    rgbfeaxy[:, :, c1+c2+1] = (1/max(h, w)) * y.T
    return rgbfeaxy

def project_onto_rgb_hull( RGB_palette, data, verbose=True ):
    ### Given: RGB simplified convex hull, all data
    ### Return: all projected data onto RGB hull if they are outside the hull
    if verbose:
        print( 'Project pixels outside onto RGB hull.' )
    hull = ConvexHull( RGB_palette ) 
    tri = Delaunay( RGB_palette )
    simplices = tri.find_simplex( data, tol=1e-6 )
    
    new_data = np.copy( data )
    for i in range( data.shape[0] ):
        if simplices[i] < 0:
            dist_list=[]
            for j in range( hull.simplices.shape[0] ):
                result = DCPPointTriangle( new_data[i], hull.points[hull.simplices[j]] )
                dist_list.append( result )
            sort_dist = sorted( dist_list, key=lambda d: d['distance'] )
            new_data[i] = sort_dist[0]['closest']
    if verbose:
        print( 'Done projection.' )
    return new_data

def Star_coordinates( vertices, data, space='RGB' ):
    ## Find the star vertex
    star = np.argmin( np.linalg.norm( vertices, axis=1 ) ) 
    ## Make a mesh for the palette 
    hull = ConvexHull( vertices ) 
    ## Star tessellate the faces of the convex hull 
    simplices = [ [star] + list(face) for face in hull.simplices if star not in face ] 
    barycoords = -1*np.ones( ( data.shape[0], len(vertices) ) ) 
    
    if space == 'AB':
        data = project_onto_ab_hull( vertices, data )
    else:
        data = project_onto_rgb_hull( vertices, data )
    
    ## Barycentric coordinates for the data in each simplex 
    for s in simplices:
        s0 = vertices[s[:1]]
        # if ill-conditioned, then solve it in LS
        if np.isinf( np.linalg.cond( (vertices[s[1:]]-s0).T ) ):
            b = np.linalg.lstsq( (vertices[s[1:]]-s0).T, (data-s0).T, rcond=None )[0].T
            '''
            rhs_size = ((data-s0).T).shape[1]
            b = np.zeros( ( rhs_size, 3 ) )
            for i in range( rhs_size ):
                b[i, :] = nnls( (vertices[s[1:]]-s0).T, (data-s0).T[:,i] )[0]
            '''
        else:
            b = np.linalg.solve( (vertices[s[1:]]-s0).T, (data-s0).T ).T 
            #b = np.linalg.solve( (vertices[s[1:]]-s0).T, (data-s0).T ).T 
        b = np.append( 1-b.sum(axis=1)[:,None], b, axis=1 ) 
        ## Update barycoords whenever data is inside the current simplex (with threshold). 
        mask = (b>=-1e-8).all(axis=1) 
        barycoords[mask] = 0. 
        barycoords[np.ix_(mask,s)] = b[mask] 
    return barycoords 
 
def Delaunay_coordinates( vertices, data ): # Adapted from Gareth Rees 
    data = project_onto_rgb_hull( vertices, data, False )
    # Compute Delaunay tessellation. 
    tri = Delaunay( vertices ) 
    # Find the tetrahedron containing each target (or -1 if not found). 
    simplices = tri.find_simplex(data, tol=1e-6) 
    assert (simplices != -1).all() # data contains outside vertices. 
    # Affine transformation for simplex containing each datum. 
    X = tri.transform[simplices, :data.shape[1]] 
    # Offset of each datum from the origin of its simplex. 
    Y = data - tri.transform[simplices, data.shape[1]] 
    # Compute the barycentric coordinates of each datum in its simplex. 
    b = np.einsum( '...jk,...k->...j', X, Y ) 
    barycoords = np.c_[b,1-b.sum(axis=1)] 
    # Return the weights as a sparse matrix. 
    rows = np.repeat(np.arange(len(data)).reshape((-1,1)), len(tri.simplices[0]), 1).ravel() 
    cols = tri.simplices[simplices].ravel() 
    vals = barycoords.ravel() 
    return coo_matrix( (vals,(rows,cols)), shape=(len(data),len(vertices)) ).tocsr()

def rgbfeaxy_pca( rgbfeaxy, dim=5, normalize=True ):
    rgbfeaxy = rgbfeaxy - np.mean( rgbfeaxy, axis=0 )
    covar = rgbfeaxy.T @ rgbfeaxy     # compute covariance
    eigval, eigvec = np.linalg.eig( covar )   # solve for eigenvectors/eigenvalues
    sort_ind = np.flip( np.argsort( eigval ) )   # sort to find eigenvalues in descending order
    sort_eigvec = eigvec[:, sort_ind]
    proj = sort_eigvec[:, :dim]     # find projection matrix (first `dim` largest eigenvectors of covariance)
    pca_rgbfeaxy = np.ascontiguousarray(rgbfeaxy) @ np.ascontiguousarray(proj)     # project to lower dimension
    if normalize:
        # normalize each channel to [0,1]
        for i in range( pca_rgbfeaxy.shape[1] ):
            pca_rgbfeaxy[:, i] -= np.min( pca_rgbfeaxy[:, i] )
            pca_rgbfeaxy[:, i] /= np.max( pca_rgbfeaxy[:, i] )
    return pca_rgbfeaxy

def proj_rgbfeaxy_to_rgbxy( rgbfeaxy ):
    rgbxy = np.zeros( (rgbfeaxy.shape[0],5) )
    rgbxy[:, :3] = rgbfeaxy[:, :3]
    rgbxy[:, 3:] = rgbfeaxy[:, rgbfeaxy.shape[1]-2:]
    return rgbxy

def RGBXY_weights( RGB_palette, RGBXY_data, RGBFEAXY=None, *mask ):
    '''
    Given:
        `RGB_palette`: a k-by-3 numpy array, where k is number of palette
        `RGBXY_data`: a n-by-5 numpy array, where n is number of pixels
        `RGBFEAXY`: a n-by-133 numpy array, where n is number of pixels
        `mask`: first component is image shape (h,w); second component is  
                       binary mask with shape (h,w) indicating local region
    Retrun:
        A n-by-k weights.
    '''
    RGBXY_hull_vertices = RGBXY_data[ ConvexHull( RGBXY_data ).vertices ]
    
    # if rgbfeaxy (133-dimensional space) is given, we find approximate convexhull
    if RGBFEAXY is not None:
        pca_rgbfeaxy = rgbfeaxy_pca( RGBFEAXY, dim=5 )
        if len( mask ) != 0:
            image_shape = mask[0]
            pca_rgbfeaxy = pca_rgbfeaxy.reshape( ( image_shape[0], image_shape[1], 5 ) )
            pca_rgbfeaxy = pca_rgbfeaxy[ mask[1] ]
        print( 'Compute convexhull in PCA lower dimensional space...' )
        rgbfeaxy_hull_vertices = RGBFEAXY[ ConvexHull( pca_rgbfeaxy ).vertices ]
        dom_rgbxy_vertices = proj_rgbfeaxy_to_rgbxy( rgbfeaxy_hull_vertices )
        RGBXY_hull_vertices = np.unique( np.concatenate( ( dom_rgbxy_vertices, RGBXY_hull_vertices ) ), axis=0 )
    
    print( 'Compute generalized barycentric coordinates...' )
    W_RGBXY = Delaunay_coordinates( RGBXY_hull_vertices, RGBXY_data )
    #W_RGB = Delaunay_coordinates( RGB_palette, RGBXY_hull_vertices[:,:3] ).todense()
    W_RGB = Star_coordinates( RGB_palette, RGBXY_hull_vertices[:,:3] )
    return W_RGBXY.dot( W_RGB )
