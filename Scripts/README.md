# Auxiliar Scripts

This folder has auxiliar scripts to execute the image processing and machine learning algorithms

### Scripts
    * create_neg_samples.py
        Creates negative samples cutting the frame of video into squares or rectangles in dimensions defined previously. 
    * resize_training_images.py
        Resizes the training images to the same dimension but the images below 35x35 are eliminated.
    * script_GT.py
        Reads the ground truth file(s) and remove the lines where the image dimension are inferior to 48x48.