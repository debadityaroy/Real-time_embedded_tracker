# Real-time_embedded_tracker
A real-time tracker (around 16 fps) for aerial videos for NVIDIA Jetson TX-1

Built over the existing tracker

Hierarchical Convolutional Features for Visual Tracking (ICCV 2015)
https://github.com/jbhuang0604/CF2

Usage:
python run_tracker.py video_name

Directions for creating input:

1. Convert video into frames.
2. Create a folder in data directory with name as video_name.
3. Create a folder in video_name directory called img
4. Move all images into img directory ( .jpg or .png)
5. Create a grountruth_rect.txt file in video_name directory.
6. Enter a single line in groundtruth_rect.txt as follows:
      x y w h 
      where x = leftmost corner x position
            y = leftmost corner y position
            w = width of object
            h = height of object
        
Additional files:
Keep in main folder

Pretrained CNN model: VGG_CNN_S_deploy.prototxt
https://gist.github.com/ksimonyan/fd8800eeb36e276cd6f9#file-vgg_cnn_s_deploy-prototxt

Pretrained CNN prototxt : VGG_CNN_S.caffemodel
http://www.robots.ox.ac.uk/~vgg/software/deep_eval/releases/bvlc/VGG_CNN_S.caffemodel

        
Output:

The tracking output will be shown in a window and frames per second 
will be printed at the end of processing the entire video.
