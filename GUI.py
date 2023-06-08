import sys
import os
import scipy.io as sio

import numpy as np
import cv2
from PIL import Image

from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
from qtwidgets import Toggle

from sparse import Edits
from func.seg import *

# In command line: "pip install opencv-python-headless" to avoid qt complaining two set of binaries

### This sometimes happens: 'qt.gui.icc: fromIccProfile: failed minimal tag size sanity'

## A workaround for a bug in Qt with Big Sur. Later Qt versions don't need this workaround.
## I don't know which version.
import platform
if platform.system() == 'Darwin' and platform.mac_ver()[0] >= '10.16':
    os.environ['QT_MAC_WANTS_LAYER'] = '1'
    
class MainWindow( QWidget ):
    def __init__( self ):
        super().__init__()
        
        self.EDIT = Edits()
        
        self.setWindowTitle( 'Sparse Editing' )
        self.setGeometry( 100, 100, 150, 100 )    # (x, y, width, height)
        
        # UI initialization
        self.checkboxes()
        self.labels()
        self.textlabels()
        self.boxes()
        self.colordialogs()
        self.buttons()
        self.comboboxes()
        self.separated_lines()
        
        # layout setup
        self.widgets_setup()
        self.layout_setup()
        
    def checkboxes( self ):
        ## clipping checkbox
        self.palette_cons_check = QCheckBox()
        self.palette_cons_check.setMinimumWidth(120)
        self.palette_cons_check.setMaximumWidth(120)
    
    def colordialogs( self ):
        ## color dialogs for clicking on palettes and selected pixel constraint
        self.color_diag_palette = QColorDialog()
        self.color_diag_image = QColorDialog()
        self.color_diag_control = QColorDialog()
        
    def labels( self ):
        ## image panel
        self.imagePanel = QLabel()
        self.makeClickable( self.imagePanel )
        
        ## palettes panel
        self.palettesPanel = QLabel()
        self.makeClickable_control( self.palettesPanel )
        
        ## panel for selected pixel
        self.pixelPanel_selected = QLabel()
        self.pixelPanel_selected.setAlignment( Qt.AlignCenter )
        self.pixelPanel_selected.setStyleSheet( "background-color: white" ) 
        
        ## panel for changed pixel
        self.pixelPanel_changed = QLabel()
        self.pixelPanel_changed.setAlignment( Qt.AlignCenter )
        self.pixelPanel_changed.setStyleSheet( "background-color: black" ) 
        self.makeClickable( self.pixelPanel_changed )
        
    def textlabels( self ):
        self.palette_size_spin_box_text = QLabel( 'Palette size: ' )
        
        ## selected and changed pixel color
        self.selected_color_text = QLabel( 'Selected color: ' )
        self.select_color_original_text = QLabel( 'Original ' )
        self.select_color_original_text.setAlignment( Qt.AlignCenter )
        self.select_color_new_text = QLabel( 'New ')
        self.select_color_new_text.setAlignment( Qt.AlignCenter )
        
        ## option section: clipping method
        self.option_text = QLabel( 'Options: ' )
        self.palette_cons_check_text = QLabel( 'Place palette constraint: ' )
    
    def boxes( self ):
        self.load_everything = QVBoxLayout()   # box for image loading, global palette computation, and global weight computation
        self.palette_size_box = QHBoxLayout()  # box for spin box and its text for # of palette to be extracted
        self.save_image_box = QHBoxLayout()  # box for saving currently edited image
        
        self.select_color_original_box = QVBoxLayout()     # box for selected pixel color
        self.select_color_new_box = QVBoxLayout()   # box for changed pixel color
        self.all_select_color_box = QHBoxLayout()   # box for all selected color boxes
        
        #self.palette_cons_check_box = QHBoxLayout()  # option for placing palette constraintcheck box
        
    def buttons( self ):
        ### auxilaries for button setup
        def set_button_utils( button, func, text, width ):
            button.clicked.connect( func )
            button.setToolTip( text )
            
            if len( width ) == 1:
                button.setMaximumWidth( width[0] )
            else:
                button.setMinimumWidth( width[0] )
                button.setMaximumWidth( width[1] )
                
        ## button for selecting an input image
        self.img_btn = QPushButton( 'Load Image...' )
        set_button_utils( self.img_btn, self.get_image, 'Press the button to <b>select</b> an image.', (120,) )
        
        ## button for selecting an input image
        self.features_btn = QPushButton( 'Load Features...' )
        set_button_utils( self.features_btn, self.load_features, 'Press the button to <b>select</b> the features of input image.', (120,) )
        
        ## button for reseting palette
        self.reset_select_color = QPushButton( 'Reset' )
        set_button_utils( self.reset_select_color, self.reset_specific_cons, 'Press the button to <b>reset</b> the palette.', (120,) )
        
        self.delete_select_color = QPushButton( 'Delete' )
        set_button_utils( self.delete_select_color, self.delete_specific_cons, 'Press the button to <b>delete</b> previous color.', (120,) )

        ## button for directly forcing local edits
        self.localize_btn = QPushButton( 'Localize' )
        set_button_utils( self.localize_btn, self.localize_cons, 'Press the button to <b>localize</b> your selected edit.', (100,) )
        
        ## button for baking any change
        self.edit_satisfy = QPushButton( 'Bake' )
        set_button_utils( self.edit_satisfy, self.satisfy_current_edits, 'Press the button to <b>bake-in</b> any edit.', (100,) )
        
        ## button for undoing previous edits (until previous baked point)
        self.undo_satisfy = QPushButton( 'Undo' )
        set_button_utils( self.undo_satisfy, self.reset_to_previous_edit, 'Press the button to <b>go to previous</b> edit.', (100,) )
        
        ## button for saving edited image
        self.save_btn = QPushButton( 'Save Image' )
        set_button_utils( self.save_btn, self.save_recolor_image, 'Press the button to <b>save</b> the image.', (100,100) )
        
        ## button for saving current palette
        self.save_palette_btn = QPushButton( 'Save Palette' )
        set_button_utils( self.save_palette_btn, self.save_palette, 'Press the button to <b>save</b> the palette.', (100,100) )
        
        ## button for loading segmentation along with local palettes 
        self.load_seg_btn = QPushButton( 'Load Tree...' )
        set_button_utils( self.load_seg_btn, self.load_seg, 'Press the button to <b>load</b> your designed hierarchical segmentation.', (130,130) )
        
        ## button for saving palette tree
        self.save_palette_tree_btn = QPushButton( 'Save Palette Tree' )
        set_button_utils( self.save_palette_tree_btn, self.save_palette_tree, 'Press the button to <b>save</b> palettes in the hierarchy in json file.', (130,130) )
        
    def comboboxes( self ):
        ## creating spin box
        self.palette_size_spin_box = QSpinBox()
        self.palette_size_spin_box.setMaximumWidth( 50 )
        self.palette_size_spin_box.setValue(6)
    
    def separated_lines( self ):
        self.line1 = QFrame()
        self.line1.setFrameShape( QFrame.HLine )
        self.line1.setFrameShadow( QFrame.Raised )
        self.line1.setLineWidth(3)
        
        self.line2 = QFrame()
        self.line2.setFrameShape( QFrame.HLine )
        self.line2.setFrameShadow( QFrame.Raised )
        self.line2.setLineWidth(3)
    
    def widgets_setup( self ):
        ## first section: image loading (along with palette and weight computation), save image/palette buttons
        self.palette_size_box.addWidget( self.palette_size_spin_box_text )
        self.palette_size_box.addWidget( self.palette_size_spin_box )
        self.palette_size_box.addStretch(30)
        
        self.load_everything.addWidget( self.img_btn )
        self.load_everything.addLayout( self.palette_size_box )
        self.load_everything.addWidget( self.features_btn )
        self.load_everything.addWidget( self.load_seg_btn )
        
        self.save_image_box.addWidget( self.save_btn )
        self.save_image_box.addWidget( self.save_palette_btn )
        self.save_image_box.addStretch(30)
        
        ## second section: pixel constraint with its original color and changed color
        self.select_color_original_box.addWidget( self.select_color_original_text )
        self.select_color_original_box.addWidget( self.pixelPanel_selected )
        self.select_color_original_box.addWidget( self.reset_select_color )
        self.select_color_original_box.addStretch(30)
        
        self.select_color_new_box.addWidget( self.select_color_new_text )
        self.select_color_new_box.addWidget( self.pixelPanel_changed )
        self.select_color_new_box.addWidget( self.delete_select_color )
        self.select_color_new_box.addStretch(30)
    
        self.all_select_color_box.addLayout( self.select_color_original_box )
        self.all_select_color_box.addLayout( self.select_color_new_box )
        self.all_select_color_box.addStretch(30)
        
        ## third section: option section (clipping method)
        #self.palette_cons_check_box.addWidget( self.palette_cons_check_text )
        #self.palette_cons_check_box.addWidget( self.palette_cons_check )
        #self.palette_cons_check_box.addStretch(30)
    
    def layout_setup( self ):
        grid = QGridLayout()
        grid.setSpacing(12)
        
        grid.addLayout( self.load_everything, 0, 0 )
        grid.addWidget( self.imagePanel, 0, 1, 15, 1 )
        grid.addWidget( self.palettesPanel, 0, 2, 15, 1 )
        grid.addLayout( self.save_image_box, 1, 0 )
        grid.addWidget( self.line1, 2, 0 )
        grid.addWidget( self.selected_color_text, 3, 0 )
        grid.addLayout( self.all_select_color_box, 4, 0 )
        grid.addWidget( self.localize_btn, 6, 0 )
        grid.addWidget( self.undo_satisfy, 7, 0 )
        grid.addWidget( self.edit_satisfy, 8, 0 )
        grid.addWidget( self.line2, 9, 0 )
        grid.addWidget( self.option_text, 10, 0 )
        #grid.addLayout( self.palette_cons_check_box, 11, 0 )
        #grid.addWidget( self.load_seg_btn, 11, 0 )
        grid.addWidget( self.save_palette_tree_btn, 12, 0 )
        self.setLayout(grid)
        
        self.show()


    ############################################################################################################################
    ### Set image (with constraints if there are some) on the image panel
    def set_image( self, panel, image, constraints=[] ):
        #Load the image into the label
        height, width, dim = image.shape
        image = np.asarray((image*255.).round().clip(0, 255).astype(np.uint8))
        
        qim = QImage( image.data, width, height, 3 * width, QImage.Format_RGB888 )
        if constraints == []:
            panel.setPixmap( QPixmap( qim ) )
            panel.repaint()
        else:
            ### adopt from: https://stackoverflow.com/questions/59866185/how-to-draw-with-qpainter-on-top-of-already-placed-qlabel-or-qpixmap
            pixmap = QPixmap( qim )
            qp = QPainter( pixmap )
            for i in range( len(constraints) ):
                x, y = constraints[i][0][1], constraints[i][0][0]
                
                # black outer
                pen = QPen( Qt.black, 2 )
                qp.setPen( pen )
                qp.drawEllipse(x-2, y-2, 10, 10)
                
                # white fille
                pen = QPen( Qt.white, 2 )
                qp.setPen( pen )
                qp.drawEllipse(x, y, 6, 6)
                
                # black interior
                pen = QPen( Qt.black, 2 )
                qp.setPen( pen )
                qp.drawEllipse(x+2,y+2, 2, 2)
                
            qp.end()
            panel.setPixmap( pixmap )
            panel.repaint()
            
    ### Function for loading an input image
    def get_image( self ):
        img = QFileDialog.getOpenFileName( self, 'Select file' )
        if img:
            path = img[0]
            self.EDIT.imagePath = path
            print ( "Loading Image..." )
            
            # load image with numpy array
            self.EDIT.image = cv2.cvtColor( cv2.imread( path ), cv2.COLOR_BGR2RGB ) / 255.
            self.EDIT.image_og = self.EDIT.image
            self.EDIT.image_frozen = self.EDIT.image
            
            # set original image onto image panel
            self.imagePanel.setPixmap( QPixmap( path ) )
            
            # extracting palette
            self.EDIT.extract_palette( self.palette_size_spin_box.value() )
            
            # paste it 
            self.set_image( self.imagePanel, self.EDIT.image )
            
            '''
            # hard-code features
            print( "Loading features..." )
            
            # load features with numpy array
            self.EDIT.features = sio.loadmat( 'features/nesi-128.mat' )['features']
            print( "Features loaded." )
            
            # use features to compute our new weight scheme
            self.EDIT.extract_weights()
            
            
            # hard-code seg
            print( 'Load tree...' )
            import func.seg2tree as seg2tree
            tree = seg2tree.get_user_tree( 'sss_nesi' )
            
            # preparing for palette tree information
            global_weights = self.EDIT.per_pixel_weights.reshape( ( self.EDIT.image.shape[0], self.EDIT.image.shape[1], self.palette_size_spin_box.value() ) )
            self.EDIT.ptree_process = Palette_tree( tree.labels, tree.parent2child, tree.constraints_tracker, self.EDIT.image, self.EDIT.palette, global_weights )
            load_for_test( self.EDIT.ptree_process )
            
            #self.ptree_og = copy_tree(self.ptree)   # keep the original ptree
            self.EDIT.ptree_baseline = copy_tree( self.EDIT.ptree_process )   # keep an copy ptree for baseline optimization
            self.EDIT.ptree_reset = copy_tree( self.EDIT.ptree_process )   # keep the reset ptree
            
            # set up control panel visualization
            self.EDIT.palette_constraints = tree.constraints_tracker
            self.update_control_panel()
            
            # use image from the original palette tree to paste image onto panel
            self.EDIT.image = self.EDIT.ptree_process.image
            self.set_image( self.imagePanel, self.EDIT.image )
            
            print( 'Tree constructed!' )
            '''
            
        else:
            QMessageBox.warning( self, 'Warning' , 'No file selected.' )
    
    ### Function for loading features for the input image
    def load_features( self ):
        features = QFileDialog.getOpenFileName( self, 'Select file' )
        if features:
            path = features[0]
            print( "Loading features..." )
            
            # load features with numpy array
            self.EDIT.features = sio.loadmat( path )['features']
            print( "Features loaded." )
            
            # use features to compute our new weight scheme
            self.EDIT.extract_weights()
        else:
            QMessageBox.warning( self, 'Warning' , 'No file selected.' )
        
    ### Loading segmentation tree for hierarchical editing
    def load_seg( self ):
        import func.seg2tree as seg2tree
        file = QFileDialog.getOpenFileName( self, 'Select file' )
        
        if file:
            print( 'Load tree...' )
            folder_path = file[0].split( '/' )[-2] + '/'
            tree = seg2tree.get_user_tree( folder_path )
            
            # preparing for palette tree information
            global_weights = self.EDIT.per_pixel_weights.reshape( ( self.EDIT.image.shape[0], self.EDIT.image.shape[1], self.palette_size_spin_box.value() ) )
            self.EDIT.ptree_process = Palette_tree( tree.labels, tree.parent2child, tree.constraints_tracker, self.EDIT.image, self.EDIT.palette, global_weights )
            load_for_test( self.EDIT.ptree_process )
            
            #self.ptree_og = copy_tree(self.ptree)   # keep the original ptree
            self.EDIT.ptree_baseline = copy_tree( self.EDIT.ptree_process )   # keep an copy ptree for baseline optimization
            self.EDIT.ptree_reset = copy_tree( self.EDIT.ptree_process )   # keep the reset ptree
            
            # set up control panel visualization
            self.EDIT.palette_constraints = tree.constraints_tracker
            self.update_control_panel()
            
            # use image from the original palette tree to paste image onto panel
            self.EDIT.image = self.EDIT.ptree_process.image
            self.set_image( self.imagePanel, self.EDIT.image )
            
            print( 'Tree constructed!' )
        else:
            QMessageBox.warning( self, 'Warning' , 'No hierarchy selected.' )
    
    def save_palette_tree( self ):
        import json
        
        with open( 'palette-tree.json', 'w' ) as f:
            json.dump( {node: palette.tolist() for node, palette in self.EDIT.ptree_process.node2palette.items()}, f )
            
        print( 'Palette tree saved.' )
        
    ############################################################################################################################
    ### update palette control panel based on current palette tree
    def update_control_panel( self ):
        # set up control panel visualization
        labeling = self.EDIT.ptree_process.find_current_labeling()
        self.EDIT.composite = self.EDIT.ptree_process.get_composited_image_and_palettes( self.EDIT.image_og, labeling )
        
        height, width, dim = self.EDIT.composite.shape
        qim = QImage( self.EDIT.composite.data, width, height, 3 * width, QImage.Format_RGB888 )
        #self.palettesPanel.setPixmap( QPixmap( qim ) )
        
        ### adopt from: https://stackoverflow.com/questions/59866185/how-to-draw-with-qpainter-on-top-of-already-placed-qlabel-or-qpixmap
        pixmap = QPixmap( qim )
        qp = QPainter( pixmap )
        
        for node in self.EDIT.palette_constraints:
            for con in self.EDIT.palette_constraints[ node ]:   # format: [ (local_index, location, palette_color) ]
                x, y = con[1][1], con[1][0]
                
                # black outer
                pen = QPen( Qt.black, 2 )
                qp.setPen( pen )
                qp.drawEllipse(x-2, y-2, 10, 10)
                
                # white filler
                pen = QPen( Qt.white, 2 )
                qp.setPen( pen )
                qp.drawEllipse(x, y, 6, 6)
                
                # black interior
                pen = QPen( Qt.black, 2 )
                qp.setPen( pen )
                qp.drawEllipse(x+2,y+2, 2, 2)
            
        qp.end()
        self.palettesPanel.setPixmap( pixmap )
        self.palettesPanel.repaint()
    
    ### update both panel with current constraints and edits
    def update_image_and_palette_control_panel( self, tree ):
        #recon_img = np.zeros( self.EDIT.image.shape )
        recon_img = np.copy( self.EDIT.image )
        tree.reconstruct_image_from_leaves( 0, recon_img )
        tree.image = recon_img.clip( 0, 1 )
        
        # paste edited image
        self.EDIT.image = tree.image
        self.set_image( self.imagePanel, self.EDIT.image, self.EDIT.pixel_constraints )
        
        # update control panel panel
        self.update_control_panel()
    
    ############################################################################################################################
    ### baked in all the changes
    def satisfy_current_edits( self ):
        # bake in current changes first
        self.EDIT.ptree_baseline = copy_tree( self.EDIT.ptree_process )
        self.EDIT.image = self.EDIT.ptree_baseline.image
        self.EDIT.image_og = self.EDIT.ptree_baseline.image
        self.release_all_active_constraints()
        
        # if baked, bake latest changes to baseline
        self.update_image_and_palette_control_panel( self.EDIT.ptree_baseline )
    
    ### reset every edit to previous baseline palette tree
    def reset_to_previous_edit( self ):
        # reset to baseline hierarchy
        self.EDIT.ptree_process = copy_tree( self.EDIT.ptree_baseline )
        self.EDIT.image = self.EDIT.ptree_baseline.image
        self.EDIT.image_og = self.EDIT.ptree_baseline.image
        self.release_all_active_constraints()
        
        # if baked, bake latest changes to baseline
        self.update_image_and_palette_control_panel( self.EDIT.ptree_baseline )
            
    ### adding image-space constraint
    def add_pixel_constraint( self, pos ):
        # find color
        
        # use `frozen` version of image to ensure the optimization always starts from scratch
        color = np.asarray( ( self.EDIT.image_frozen[ pos ]*255. ).round().clip( 0, 255 ).astype( np.uint8 ) ) 
        
        # use average colors from a 3x3 window
        region_colors = self.EDIT.image[ pos[0]-1: pos[0]+1, pos[1]-1: pos[1]+1 ]
        target_color = np.asarray( ( np.mean( region_colors.reshape( -1, 3 ), axis = 0 )*255. ).round().clip( 0, 255 ).astype( np.uint8 ) )
        print( 'Adding color constraint at ', pos, ' with color: ', target_color )
        
        # use `False` as default localized option
        localize = False
        color_constraint = ( pos, [color / 255., target_color / 255.], [localize] )
        self.EDIT.pixel_constraints.append( color_constraint )
        
        # paste selected color on both original and new color panels
        color_str = '#%02x%02x%02x' % (color[0], color[1], color[2])
        self.pixelPanel_selected.setStyleSheet( "background-color: " + color_str )
        self.pixelPanel_changed.setStyleSheet( "background-color: " + color_str )
        self.set_image( self.imagePanel, self.EDIT.image, self.EDIT.pixel_constraints )
        
        # call the optimizer (always use baseline hierarchy)
        self.EDIT.ptree_process, _ = self.EDIT.sparse_edit_handler( self.EDIT.ptree_baseline )
        
        # render result
        self.update_image_and_palette_control_panel( self.EDIT.ptree_process )
        print( '**Current color constraints: ', self.EDIT.ptree_process.constraints_tracker )
    
    
    def add_palette_constraint( self,  cons ):
        # `cons` is a tuple: (  node, palette index, location, [target color] )
        # palette constraints format: {0: [  (palette color index, location, [ selected change ]) ], 1: [  (palette color index, location, [ selected change ]) ], ...}
        self.EDIT.palette_constraints[ cons[0] ].append( ( cons[1], cons[2], cons[3] ) )
        print( 'Adding palette constraint at node', cons[0], ' at ', cons[1], ' \'s palette color with target: ', cons[3][0] )
        
        # paste selected color on both original and new color panels
        color = ( cons[3][0] ).round().clip( 0, 255 ).astype( np.uint8 )
        color_str = '#%02x%02x%02x' % (color[0], color[1], color[2])
        self.pixelPanel_selected.setStyleSheet( "background-color: " + color_str )
        self.pixelPanel_changed.setStyleSheet( "background-color: " + color_str )
        
        # update palette panel
        self.update_control_panel()
        
        # call optimizer
        self.EDIT.ptree_process, _ = self.EDIT.sparse_edit_handler( self.EDIT.ptree_baseline )
        
        # render result
        self.update_image_and_palette_control_panel( self.EDIT.ptree_process )
        print( '**Current palette constraints: ', self.EDIT.palette_constraints )
        
        
    ### reset specific image-space constraint
    def reset_specific_cons( self ):
        ####
        #### TODO: call the optimizer!
        #### This method might be redundant. (This method should be implemented as reset everything.)
        
        # bake in current changes first
        self.reset_or_bake( 'baked' )
        
        # reset to original color at that pixel
        if self.EDIT.user_colors != []:
            self.EDIT.user_colors[ self.EDIT.ind_pixel ] = self.EDIT.image[ self.EDIT.user_pos[ self.EDIT.ind_pixel ] ]
    
    ### delete specific image-space constraint
    def delete_specific_cons( self ):
        if self.EDIT.image_click:
            # delete constraint on user's click
            self.EDIT.pixel_constraints.pop( self.EDIT.ind_pixel ) 
            print( 'Delete the color constraint.' )
            
            # call the optimizer (always use baseline hierarchy)
            self.EDIT.ptree_process, _ = self.EDIT.sparse_edit_handler( self.EDIT.ptree_baseline )
            
            # render result
            self.update_image_and_palette_control_panel( self.EDIT.ptree_process )
        
        if self.EDIT.palette_click:
            self.EDIT.palette_constraints[ self.EDIT.node_local ].pop( self.position_2_user_clicks_ind_palette() )
            print( 'Delete the palette constraint.' )
            
            # call the optimizer
            self.EDIT.ptree_process, _ = self.EDIT.sparse_edit_handler( self.EDIT.ptree_baseline )
            
            # render result
            self.update_image_and_palette_control_panel( self.EDIT.ptree_process )
            
            # update palette panel
            self.update_control_panel()
            print( '**Current palette constraints: ', self.EDIT.palette_constraints )
        
    ### indicator to localize the edit with respect to chosen constraint
    def localize_cons( self ):
        self.EDIT.pixel_constraints[ self.EDIT.ind_pixel ][2][0] = True
    
    ############################################################################################################################
    ### clickable function in palette control panel to recolor global/local palettes
    def makeClickable_control( self, widget ):
        def SendClickSignal( widget, evnt ):
                # click indicator for different panels
                self.EDIT.palette_click = True
                self.EDIT.image_click = False
                
                # clicked information
                loc = ( evnt.pos().y(), evnt.pos().x() )
                self.palette_pick_color = self.EDIT.composite[ loc ] 
                self.EDIT.node_local = self.EDIT.ptree_baseline.find_node_from_current_constraints_on_click( loc )   # find region w.r.t. user's click
                
                # use process hierarchy to find placed click
                local_palette = self.EDIT.ptree_process.node2palette[ self.EDIT.node_local ]        # find corresponding local palette and index for user's click
                self.EDIT.ind_local = np.argmin( np.linalg.norm( local_palette - self.palette_pick_color/255., axis=1 ) )   # find which palette color that user clicks
                cons = ( self.EDIT.node_local, self.EDIT.ind_local, loc, [ self.palette_pick_color ] )
                
                #self.EDIT.ind_palette = -1  # initialize to -1
                self.EDIT.ind_palette = self.position_2_user_clicks_ind_palette()
                
                if self.EDIT.ind_palette == -1:   # if we did not find same previous click, then we add new constraint
                    self.add_palette_constraint( cons )
                else: # paste the previous clicked palette color onto the panel     # is it enough?
                    color_og = np.asarray( ( self.palette_pick_color ).round().clip( 0, 255 ).astype( np.uint8 ) )
                    color_new = np.asarray( ( self.palette_pick_color ).round().clip( 0, 255 ).astype( np.uint8 ) )
                
                    color_str_original = '#%02x%02x%02x' % ( color_og[0], color_og[1], color_og[2] )
                    color_str_new = '#%02x%02x%02x' % ( color_new[0], color_new[1], color_new[2] )
                
                    self.pixelPanel_selected.setStyleSheet( "background-color: " + color_str_original )
                    self.pixelPanel_changed.setStyleSheet( "background-color: " + color_str_new )
        
        widget.emit( SIGNAL( 'clicked()' ) )
        widget.mousePressEvent = lambda evnt: SendClickSignal( widget, evnt )        
    
    ### clickable function for image panel and adding/deleting pixel constraint panels
    def makeClickable( self, widget ):
        def SendClickSignal( widget, evnt ):
            ## if user clicks pixel panel and wish to change color of that pixel, pop up color dialog
            if widget == self.pixelPanel_changed:
                self.select_color()
            
            ## if user clicks on image panel, it means that they want to add constraint or select constraint to change pixel color
            if widget == self.imagePanel:
                # click indicator for different panels
                self.EDIT.image_click = True
                self.EDIT.palette_click = False
                
                self.changed_image_x = int( evnt.position().x() )
                self.changed_image_y = int( evnt.position().y() )
                cur_pos = (self.changed_image_y, self.changed_image_x)
                
                self.EDIT.ind_pixel = -1  # initialize to -1
                
                # if there are some constraints
                if len(self.EDIT.pixel_constraints) != 0:
                    self.EDIT.ind_pixel = self.position_2_user_clicks_ind_pixel( cur_pos )
                    if self.EDIT.ind_pixel == -1:   # if we did not find same previous click, then we add new constraint
                        self.add_pixel_constraint( cur_pos )
                    else: # paste the previous clicked pixel's color onto the panel     # is it enough?
                        color_og = np.asarray( ( self.EDIT.image_og[ self.EDIT.pixel_constraints[self.EDIT.ind_pixel][0] ]*255. ).round().clip( 0, 255 ).astype( np.uint8 ) )
                        color_new = np.asarray( ( self.EDIT.image[ self.EDIT.pixel_constraints[self.EDIT.ind_pixel][0] ]*255. ).round().clip( 0, 255 ).astype( np.uint8 ) )
                        
                        color_str_original = '#%02x%02x%02x' % ( color_og[0], color_og[1], color_og[2] )
                        color_str_new = '#%02x%02x%02x' % ( color_new[0], color_new[1], color_new[2] )
                        
                        self.pixelPanel_selected.setStyleSheet( "background-color: " + color_str_original )
                        self.pixelPanel_changed.setStyleSheet( "background-color: " + color_str_new )
                else:   # first click
                    self.add_pixel_constraint( cur_pos )
                
        widget.emit( SIGNAL( 'clicked()' ) )
        widget.mousePressEvent = lambda evnt: SendClickSignal( widget, evnt )
                
    ############################################################################################################################
    # Function for changing the color on specific pixel
    def select_color( self ):
        if self.EDIT.image_click:
            self.color_diag_image.open()
            self.pick_color = np.uint8( self.EDIT.pixel_constraints[ self.EDIT.ind_pixel ][1][1] * 255. )
            self.color_diag_image.setCurrentColor( QColor(self.pick_color[0], self.pick_color[1], self.pick_color[2]) )
            self.color_diag_image.blockSignals(True)
            self.color_diag_image.currentColorChanged.connect( self.sparse_edit_pixel )
            self.color_diag_image.blockSignals(False)
        if self.EDIT.palette_click:
            self.color_diag_palette.open()
            paste_color = np.uint8( self.palette_pick_color )
            self.color_diag_palette.setCurrentColor( QColor( paste_color[0], paste_color[1], paste_color[2] ) )
            self.color_diag_palette.blockSignals(True)
            self.color_diag_palette.currentColorChanged.connect( self.sparse_edit_palette )
            self.color_diag_palette.blockSignals(False)
    
    def sparse_edit_palette( self, color ):
        target_color = np.array( [color.red(), color.green(), color.blue()] ) / 255.
        
        if not (self.EDIT.prev_optimize_color == target_color).all():
            # paste color on new color label
            color_str_new = '#%02x%02x%02x' % (color.red(), color.green(), color.blue())
            self.pixelPanel_changed.setStyleSheet( "background-color: " + color_str_new )
            
            # change target color of specific constraint in palette constraints list
            self.EDIT.palette_constraints[ self.EDIT.node_local ][ self.position_2_user_clicks_ind_palette() ][2][0] = target_color * 255.
            
            # call sparse edit handler
            self.EDIT.ptree_process, split_indicator = self.EDIT.sparse_edit_handler( self.EDIT.ptree_baseline  )
            print( '***palette constraints: ', self.EDIT.palette_constraints )
            
            ### TODO: this might needs to fix!
            ### release palette constraints if there is an activated node that does not have palette constraints
            if split_indicator:
                self.EDIT.palette_constraints = { node: [] for node in self.EDIT.palette_constraints }
            
            self.update_control_panel()
            
            # render result
            self.update_image_and_palette_control_panel( self.EDIT.ptree_process )
            self.EDIT.prev_optimize_color = target_color    # update previous optimized color
    
    def sparse_edit_pixel( self, color ):
        target_color = np.array( [color.red(), color.green(), color.blue()] ) / 255.
        
        if not (self.EDIT.prev_optimize_color == target_color).all():
            # paste color on new color label
            color_str_new = '#%02x%02x%02x' % (color.red(), color.green(), color.blue())
            self.pixelPanel_changed.setStyleSheet( "background-color: " + color_str_new )
            
            # change target color of specific constraint in pixel constraints list
            self.EDIT.pixel_constraints[ self.EDIT.ind_pixel ][1][1] = target_color
            
            # call sparse edit handler
            self.EDIT.ptree_process, split_indicator = self.EDIT.sparse_edit_handler( self.EDIT.ptree_baseline  )
            print( '***color constraints: ', self.EDIT.ptree_process.constraints_tracker )
            
            
            ### release palette constraints if there is an activated node that does not have palette constraints
            if split_indicator:
                self.EDIT.palette_constraints = { node: [] for node in self.EDIT.palette_constraints }
            
            self.update_control_panel()
            
            # render result
            self.update_image_and_palette_control_panel( self.EDIT.ptree_process )
            self.EDIT.prev_optimize_color = target_color    # update previous optimized color
    
    ############################################################################################################################
    # function to save current edited image
    def save_recolor_image( self ):
        self.save_image( 1, self.EDIT.imagePath )
        
    # functions to save current image
    def save_palette( self ):
        self.save_image( 2, self.EDIT.palette )
    
    def save_image( self, option, image_path ):
        if option == 1:
            s = 'image'
            saved_img = self.EDIT.image
        else:
            s = 'palette'
            saved_img = self.EDIT.palette_img
        
        if self.EDIT.imagePath == '':
            QMessageBox.warning( self, 'Warning', 'Please select an image first.' )
        else:
            reply = QMessageBox.question( self, 'Message', "Are you sure to save your current " + s + " image on this panel?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No )
            
            if reply == QMessageBox.Yes:
                image_name = QFileDialog.getSaveFileName( self, 'Save ' + s )
                if not image_name:
                    return
                
                if image_name[0][-4:] in ['.jpg', '.png']:
                    path_name = image_name[0]
                else:
                    path_name = image_name[0] + '.png'
                
                saved_img = np.asarray((saved_img*255.).round().clip(0, 255).astype(np.uint8))
                Image.fromarray( saved_img ).save( path_name )
    
    ############################################################################################################################
    ### Helper functions
    def reset_or_bake( self, option ):
        if self.EDIT.sparse_edit_indicator:  # if it's True, it means that user just finished sparse edit, then copy the result of optimization into current ptree
            if option == 'reset':
                self.EDIT.ptree_process = copy_tree( self.EDIT.ptree )
            elif option == 'baked':
                self.EDIT.ptree = copy_tree( self.EDIT.ptree_process )
            self.EDIT.sparse_edit_indicator = False  # turn it into False since it's not sparse edit now
        
    def position_2_user_clicks_ind_pixel( self, cur_pos ):
        for i in range( len(self.EDIT.pixel_constraints) ):
            if np.linalg.norm( np.array(self.EDIT.pixel_constraints[i][0]) - np.array(cur_pos) ) <= 10:  # compromised distance to previous clicks
                return i
        return -1
    
    def position_2_user_clicks_ind_palette( self ):
        # NOTE: `self.EDIT.palette_constraints` has below format
        #  {0: [  (palette color index, location, [ selected change ]) ], 1: [  (palette color index, location, [ selected change ]) ], ...}
        for i in range( len( self.EDIT.palette_constraints[ self.EDIT.node_local ] ) ):
            cons = self.EDIT.palette_constraints[ self.EDIT.node_local ][i] 
            if self.EDIT.ind_local == cons[0]:  # if same palette color is clicked
                return i
        return -1
    
    ### clear all active constraints
    def release_all_active_constraints( self ):
        # release pixel constraints
        self.EDIT.pixel_constraints = []
        for node in self.EDIT.ptree_baseline.constraints_tracker:
            self.EDIT.ptree_baseline.constraints_tracker[ node ] = []
            
        # release palette constraints
        for node in self.EDIT.palette_constraints:
            self.EDIT.palette_constraints[ node ] = []
        
    ############################################################################################################################
    ### Function if users tend to close the app
    def closeEvent( self, event ):
        reply = QMessageBox.question( self, 'Message', "Are you sure you want to quit?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No )
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

def main():
    app = QApplication( sys.argv )
    ex = MainWindow()
    sys.exit( app.exec_() )
    
    
if __name__ == '__main__':
    main()
    
    