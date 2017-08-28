'''
% RUN_TRACKER: process a specified video using CF2
'''
from __future__ import division
import os
import re
import numpy as np
import cv2
import caffe
from matlabtb import imResample
import time
import pdb
import sys
import scipy.io

# using Pycaffe to get features
caffe.set_device(0)
caffe.set_mode_gpu()

# setup done only once

#net = caffe.Net('../VGG_ILSVRC_19_layers_deploy.prototxt','../VGG_ILSVRC_19_layers.caffemodel',caffe.TEST)
net = caffe.Net('VGG_CNN_S_deploy.prototxt','VGG_CNN_S.caffemodel',caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
mean = np.load('ilsvrc_2012_mean.npy').mean(1).mean(1)
transformer.set_mean('data', mean)
transformer.set_transpose('data', (2,0,1))
net.blobs['data'].reshape(1,3,224,224)

'''
function [img_files, pos, target_sz, ground_truth, video_path] = load_video_info(base_path, video)
%LOAD_VIDEO_INFO
%   Loads all the relevant information for the video in the given path:
%   the list of image files (cell array of strings), initial position
%   (1x2), target size (1x2), the ground truth information for precision
%   calculations (Nx2, for N frames), and the path where the images are
%   located. The ordering of coordinates and sizes is always [y, x].
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/
'''

def load_video_info(base_path,video):
    #full path to the video's files
    if (base_path[-1] != '/'):
        base_path = base_path+'/'

    video_path = base_path+video+'/'

    #try to load ground truth from text file (Benchmark's format)
    filename = video_path+'groundtruth_rect.txt'
    if not os.path.exists(filename):
        raise IOException('File does not exist: %s' % filename)
    f = open(filename,'r')
    
    #the format is [x, y, width, height]
    gt_line = f.readlines()
    ground_truth = gt_line[0].split()
    ground_truth = np.asarray(ground_truth)
    f.close()
    
    #set initial position and size
    target_sz = np.array([float(ground_truth[3]),float(ground_truth[2])])
    #print(target_sz.dtype)
    pos = np.array([float(ground_truth[1]), float(ground_truth[0])]) + np.floor(target_sz/2)
    
    ground_truth = pos

    #from now on, work in the subfolder where all the images are
    video_path = video_path+'img/'
    
    
        #general case, just list all images
    img_files_check = os.listdir(video_path)
    img_files = []
    for img in img_files_check:
        if img.endswith('.jpg') or img.endswith('.png'):
            img_files.append(img)   
    if not img_files:
        raise IOException('No image files to load.')
    img_files.sort()
    
    return  img_files, pos, target_sz, ground_truth, video_path

'''
%GAUSSIAN_SHAPED_LABELS
%   Gaussian-shaped labels for all shifts of a sample.
%
%   LABELS = GAUSSIAN_SHAPED_LABELS(SIGMA, SZ)
%   Creates an array of labels (regression targets) for all shifts of a
%   sample of dimensions SZ. The output will have size SZ, representing
%   one label for each possible shift. The labels will be Gaussian-shaped,
%   with the peak at 0-shift (top-left element of the array), decaying
%   as the distance increases, and wrapping around at the borders.
%   The Gaussian function has spatial bandwidth SIGMA.
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/


%   %as a simple example, the limit sigma = 0 would be a Dirac delta,
%   %instead of a Gaussian:
%   labels = zeros(sz(1:2));  %labels for all shifted samples
%   labels(1,1) = magnitude;  %label for 0-shift (original sample)
'''
def gaussian_shaped_labels(sigma, sz):
# function labels = gaussian_shaped_labels(sigma, sz)

    #evaluate a Gaussian with the peak at the center element
    
    rs, cs = np.meshgrid(np.asarray(range(int(sz[1]))) - np.floor(sz[1]/2)+1,np.asarray(range(int(sz[0]))) - np.floor(sz[0]/2)+1)

    ###CHGORIG 2 -added Transpose to match MATLAB output
    rs = np.transpose(rs)
    cs = np.transpose(cs)


    labels = np.exp(-0.5 / (sigma**2) * (np.power(rs,2) + np.power(cs,2)))
    
    
    
    # to convert to 8 bit double MATLAB format
    #labels[labels < 0.0001] = 0  # FIXME CHGORIG-3 not required to match MATLAB code
 
    #move the peak to the top-left, with wrap-around
    #print(np.floor(sz[0:2] / 2) - 1)
        # the roller value is equal to the absolute position of 1 in the array 
        # as 1 has to be moved to 0,0  
    roller = -labels.argmax()
    labels = np.roll(labels,roller)
    
    #dy , dx = np.unravel_index(labels.argmax(), labels.shape)
    #dy, dx = int(labels.argmax()/sz[0]), int(labels.argmax()/sz[1]) # FIXME CHGORIG-4
    #labels = np.roll(labels, (-dx, -dy), (0, 1))    # CHGORIG-4 both axis shift
    #sanity check: make sure it's really at top-left
    if labels[0,0] != 1:
        print("Gaussian labeling failed")

    return labels


# GET_SEARCH_WINDOW

def get_search_window(target_sz, im_sz, padding):
#function window_sz = get_search_window( target_sz, im_sz, padding)
    if (target_sz[0]/target_sz[1] > 2):
        # For objects with large height, we restrict the search window with padding.height
            window_sz = np.floor(target_sz*[1+float(padding['height']), 1+float(padding['generic'])])
   
    elif (np.prod(target_sz)/np.prod(im_sz[0:2]) > 0.05):
        # For objects with large height and width and accounting for at least 10 percent of the whole image,
        # we only search 2x height and width
            window_sz = np.floor(target_sz * (1 + float(padding['large'])))
    
    else:
        #otherwise, we use the padding configuration
        window_sz = np.floor(target_sz * (1 + float(padding['generic'])))
    return window_sz

# GET_FEATURES: Extracting hierachical convolutional features

def get_features(im,cos_window,layers):
#function feat = get_features(im, cos_window, layers)

    ###cv2.imshow('before', im)
    ###cv2.waitKey(5)
    #load the image in the data layer
    #im = caffe.io.load_image('cat.jpg')
    # as image is (y,x,z) we transpose it as (x,y,z)
    ###im  = np.transpose(im,[1,0,2])     #FIXME  

    ###cv2.imshow('after', im)
    ###cv2.waitKey(5)
    #im = im.astype(np.float32)
    #out = np.zeros((224,224,im.shape[2]),dtype=im.dtype)
    #imResample(im,out)
    #im = out  
    tmp_img = transformer.preprocess('data', im)
    
    '''
    img = np.transpose(tmp_img, [1, 2, 0])    

    cv2.imshow('tmp_img', img.astype(np.uint8))
    cv2.waitKey(50)
    '''

    net.blobs['data'].data[...] = tmp_img
    #compute
    net.forward()

    # other possibility : out = net.forward_all(data=np.asarray([transformer.preprocess('data', im)]))

    sz_window = cos_window.shape

    
    feat = [net.blobs['conv5'].data[0,:]]
    ''' 
            net.blobs['conv4_4'].data[0,:], 
            net.blobs['conv3_4'].data[0,:]]
    '''
    ###feat = [net.blobs['pool5'].data[0,:], 
    ###        net.blobs['pool4'].data[0,:], 
    ###        net.blobs['pool3'].data[0,:]]
    ###feat = [net.params['conv5_4'][0].data, 
    ###        net.params['conv4_4'][0].data, 
    ###        net.params['conv3_4'][0].data]

    for i in range(len(feat)):
        # Resize to sz_window
        x = feat[i] 
        # transform from (z,x,y) to (x,y,z)  ###CHGORIG     
        x = np.transpose(x,[1,2,0]) ###CHGORIG -1 changed from [2,1,0]

       
        temp = np.zeros((sz_window[0], sz_window[1], x.shape[2]), dtype=x.dtype)
	for j in range(x.shape[2]):
		feat_map = x[:,:,j]
		temp[:,:,j] = cv2.resize(feat_map,(sz_window[1],sz_window[0]))

	# using matlabbtb
        #imResample(x, temp)
        x = temp
            
        # windowing techniqe
        if cos_window.size != 0:
        # Adding dummy 3D dimension to cos_window using None
        # replicating is as mnay times as depth of x
        # perform elemen-by-element multiplication
            x = x*np.tile(cos_window[:, :, None], (1, 1, x.shape[2]))
   
        feat[i] = x
    
    return feat

'''
%GET_SUBWINDOW Obtain sub-window from image, with replication-padding.
%   Returns sub-window of image IM centered at POS ([y, x] coordinates),
%   with size SZ ([height, width]). If any pixels are outside of the image,
%   they will replicate the values at the borders.
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/
'''

def get_subwindow(im,pos,sz):
#function out = get_subwindow(im, pos, sz)

    if np.isscalar(sz):  #square sub-window
            sz = np.array([sz, sz])

    ys = np.floor(pos[0]) + np.asarray(range(int(sz[0]))) - np.floor(int(sz[0]/2))
    xs = np.floor(pos[1]) + np.asarray(range(int(sz[1]))) - np.floor(int(sz[1]/2))
    
    # convet to int
    ys = ys.astype(int)
    xs = xs.astype(int)

    # Check for out-of-bounds coordinates, and set them to the values at the borders
    ys[ ys < 0 ] = 0
    ys[ ys >= im.shape[0] ] = im.shape[0] - 1 
    xs[ xs < 0 ] = 0
    xs[ xs >= im.shape[1] ] = im.shape[1] - 1
 
    zs = np.array([0,1,2])      
 
    #extract image
    out = im[np.ix_(ys, xs, zs)]

    return out



def extractFeature(im, pos, window_sz, cos_window, indLayers):
#function feat  = extractFeature(im, pos, window_sz, cos_window, indLayers)

    # Get the search window from previous detection
    patch = get_subwindow(im, pos, window_sz)
    # Extracting hierarchical convolutional features
    feat  = get_features(patch, cos_window, indLayers)

    return feat

def predictPosition(feat, pos, indLayers, nweights, cell_size, l1_patch_num, \
    model_xf, model_alphaf):
#function pos = predictPosition(feat, pos, indLayers, nweights, cell_size, l1_patch_num, ...
#    model_xf, model_alphaf)

    # ================================================================================
    # Compute correlation filter responses at each layer
    # ================================================================================
    res_layer = np.zeros([int(l1_patch_num[0]),int(l1_patch_num[1]), len(indLayers)]);

    for ii in range(len(indLayers)):
        zf = np.fft.fft2(feat[ii], axes=(0, 1))  ###CHGORIG - Axes need to be provided as by- to match MATLAB fft2
        kzf = np.sum(zf*np.conj(model_xf[ii]), axis=2) / zf.size

        ##res_layer[:,:,ii] = np.real(np.fft.fftshift(np.fft.ifft2(model_alphaf[ii] * kzf)))  #equation for fast detection
        temp = np.real(np.fft.fftshift(np.fft.ifft2(model_alphaf[ii] * kzf)))  #equation for fast detection
        res_layer[:,:,ii]=temp/np.max(temp)     ### FIXME CHGORIG-5 normalize
        #print ii, np.max(res_layer[:,:,ii]) #, res_layer[:,:,ii]
        #print


    # Combine responses from multiple layers (see Eqn. 5)
    response = np.sum(res_layer*nweights, axis=2)
    #print 'res', response.shape, response

    # ================================================================================
    # Find target location
    # ================================================================================
    # Target location is at the maximum response. we must take into
    # account the fact that, if the target doesn't move, the peak
    # will appear at the top-left corner, not at the center (this is
    # discussed in the KCF paper). The responses wrap around cyclically.
    vert_delta, horiz_delta = np.unravel_index(response.argmax(), response.shape)
    ##print 'before vdx', vert_delta, 'hdx', horiz_delta
 
    vert_delta  = vert_delta  - np.floor(zf.shape[0]/2)
    horiz_delta = horiz_delta - np.floor(zf.shape[1]/2)
    ##print 'after  vdx', vert_delta, 'hdx', horiz_delta

    # Map the position to the image space
    movement = cell_size * np.array([vert_delta, horiz_delta])
    movement = movement.astype(int)
    pos = pos + movement
    
    return pos



def updateModel(feat, yf, interp_factor, reg_param, frame, model_xf, model_alphaf):
#function [model_xf, model_alphaf] = updateModel(feat, yf, interp_factor, reg_param, frame, ...
#    model_xf, model_alphaf)

    numLayers = len(feat)

    # ================================================================================
    # Initialization
    # ================================================================================
    xf       = []
    alphaf   = []

    # ================================================================================
    # Model update
    # ================================================================================
    for ii in range(numLayers):
            xf.append(np.fft.fft2(feat[ii], axes=(0, 1)))
            kf = np.sum(xf[ii] * np.conj(xf[ii]), axis=2) / xf[ii].size
            alphaf.append(yf/ (kf + reg_param))   # Fast training


    # Model initialization or update
    if frame == 0:  # First frame, train with a single image
            for ii in range(numLayers):
                model_alphaf.append(alphaf[ii])
                model_xf.append(xf[ii])
    
    else:   # Online model update using learning rate interp_factor
            for ii in range(numLayers):
                model_alphaf.append((1 - interp_factor) * model_alphaf[ii] + interp_factor * alphaf[ii])
                model_xf.append((1 - interp_factor) * model_xf[ii]     + interp_factor * xf[ii])
 
    
    return model_xf, model_alphaf
 
'''
% tracker_ensemble: Correlation filter tracking with convolutional features
%
% Input:
%   - video_path:          path to the image sequence
%   - img_files:           list of image names
%   - pos:                 intialized center position of the target in (row, col)
%   - target_sz:           intialized target size in (Height, Width)
%   - padding:             padding parameter for the search area
%   - reg_param:              regularization term for ridge regression
%   - output_sigma_factor: spatial bandwidth for the Gaussian label
%   - interp_factor:       learning rate for model update
%   - cell_size:           spatial quantization level
%   - show_visualization:  set to True for showing intermediate results
% Output:
%   - positions:           predicted target position at each frame
%   - time:                time spent for tracking
%
%   It is provided for educational/researrch purpose only.
%   If you find the software useful, please consider cite our paper.
%
%   Hierarchical Convolutional Features for Visual Tracking
%   Chao Ma, Jia-Bin Huang, Xiaokang Yang, and Ming-Hsuan Yang
%   IEEE International Conference on Computer Vision, ICCV 2015
%
% Contact:
%   Chao Ma (chaoma99@gmail.com), or
%   Jia-Bin Huang (jbhuang1@illinois.edu).
'''

def tracker_ensemble(video_path, img_files, pos, target_sz, padding, reg_param, \
output_sigma_factor, interp_factor, cell_size,show_visualization):
#function [positions, time] = tracker_ensemble(video_path, img_files, pos, target_sz, ...
#    padding, lambda, output_sigma_factor, interp_factor, cell_size, show_visualization)

    # ================================================================================
    # Environment setting
    # ================================================================================
    indLayers = ['conv5']#, 'conv4_4', 'conv3_4']  # The CNN layers Conv5-4, Conv4-4, and Conv3-4 in VGG Net
    ##indLayers = ['pool5', 'pool4', 'pool3']  # The CNN layers Conv5-4, Conv4-4, and Conv3-4 in VGG Net
    nweights  = np.array([1])#, 0.5, 0.25]) # FIXME  Weights for combining correlation filter responses
    numLayers = len(indLayers)

    # Get image size and search window size
    im_sz     = np.array(cv2.imread(video_path+img_files[0]).shape)
    window_sz = get_search_window(target_sz, im_sz, padding)

    # Compute the sigma for the Gaussian function label
    output_sigma = (np.sqrt(np.prod(target_sz)) * output_sigma_factor) / cell_size

    #create regression labels, gaussian shaped, with a bandwidth
    #proportional to target size    d=bsxfun(@times,c,[1 2]);

    l1_patch_num = np.floor(window_sz/ cell_size)

    # Pre-compute the Fourier Transform of the Gaussian function label
    ###yf = np.fft.fft2(gaussian_shaped_labels(output_sigma, l1_patch_num))  # ORIG
    gauss = gaussian_shaped_labels(output_sigma, l1_patch_num)
    yf = np.fft.fft2(gauss.T)   ### FIXME CHGORIG


    # Pre-compute and cache the cosine window (for avoiding boundary discontinuity)
    hanning1 = np.hanning(yf.shape[0])
    hanning2 = np.hanning(yf.shape[1])

    cos_window = np.reshape(hanning1,(hanning1.shape[0],1)) * np.reshape(hanning2,(1,hanning2.shape[0]))

    # Create video interface for visualization - TODO


    # Initialize variables for calculating FPS and distance precision
    process_time = 0
    positions = np.zeros([len(img_files), 2])
    nweights  = np.reshape(nweights,(1,1,1))

    # Note: variables ending with 'f' are in the Fourier domain.
    model_xf     = []
    model_alphaf = []

    # ================================================================================
    # Start tracking
    # ================================================================================

    for frame in range(len(img_files)):
            im = cv2.imread(video_path+img_files[frame]) # Load the image at the current frame
            if len(im.shape) == 2:
                im = np.concatenate((im[:,:,None],im[:,:,None],im[:,:,None]), axis=2)
        
    
            tic = time.time()
            # ================================================================================
            # Predicting the object position from the learned object model
            # ================================================================================
            feat = extractFeature(im, pos, window_sz, cos_window, indLayers)
            if frame > 0:   
            # Extracting hierarchical convolutional features
                ###feat = extractFeature(im, pos, window_sz, cos_window, indLayers)
                # Predict position
                pos  = predictPosition(feat, pos, indLayers, nweights, cell_size, l1_patch_num, \
                        model_xf, model_alphaf)
    
                ###print '[{}] {}'.format(frame, pos)
            print '{} {}, {}'.format(frame+1, int(pos[0]), int(pos[1]))
            # ================================================================================
            # Learning correlation filters over hierarchical convolutional features
            # ================================================================================
            # Extracting hierarchical convolutional features
            ###print 'pos', pos
            ###feat  = extractFeature(im, pos, window_sz, cos_window, indLayers);
            # Model update

            model_xf, model_alphaf = updateModel(feat, yf, interp_factor, reg_param, frame, \
                    model_xf, model_alphaf)
    
            # ================================================================================
            # Save predicted position and timing
            # ================================================================================
            positions[frame,:] = pos
            toc = time.time() - tic
            process_time = process_time + toc
            # Visualization
            if (show_visualization):

                box = np.array([pos[::-1] - target_sz[::-1]/2, target_sz[::-1]])
                box = box.astype(int)
                img_to_disp = cv2.imread(video_path+img_files[frame])
                cv2.rectangle(img_to_disp,(box[0,0],box[0,1]),(box[0,0]+box[1,0],box[0,1]+box[1,1]),(0,255,0),1)
            cv2.imshow("Tracking",img_to_disp)
            k =  0xFF & cv2.waitKey(20)
            if k==27:
                break
            #if frame == 0 : break #FIXME
  
    return positions, process_time


def run_tracker(video):
#function [fps] = run_tracker(video, show_visualization, show_plots)
    
    #path to the videos (you'll be able to choose one with the GUI).
    base_path = 'data'


    # Extra area surrounding the target as a dictionary
    padding = {'generic': 1.8, 'large': 1, 'height': 0.4}

    reg_param = 0.0001              # Regularization parameter (see Eqn 3 in our paper) changed from lambda
    output_sigma_factor = 0.1  # Spatial bandwidth (proportional to the target size)

    interp_factor = 0.01       # Model learning rate (see Eqn 6a, 6b)
    cell_size = 4              # Spatial cell size
    show_visualization = True

    #We were given the name of a single video to process.
    # get image file names, initial state, and ground truth for evaluation
    img_files, pos, target_sz, ground_truth, video_path = load_video_info(base_path, video);
    #return img_files, pos, target_sz, ground_truth, video_path
    # Call tracker function with all the relevant parameters
    positions, process_time = tracker_ensemble(video_path, img_files, pos, target_sz, padding, \
        reg_param, output_sigma_factor, interp_factor, cell_size,show_visualization)
    #print (video_path, img_files, pos, target_sz, padding, reg_param, output_sigma_factor, interp_factor, cell_size)
    fps = len(img_files) / process_time
        
    print("Frames per second :"+str(fps))       
    #return 


if __name__ == "__main__":
    #video = 'MotorRolling2'
    video = sys.argv[1]
    run_tracker(video)
