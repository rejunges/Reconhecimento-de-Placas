# Auxiliar Scripts
This folder has auxiliar scripts to execute the image processing and machine learning algorithms

### Scripts
* [create_neg_samples.py](https://github.com/rejunges/Reconhecimento-de-Placas/blob/master/Scripts/create_neg_samples.py)
	- Creates negative samples cutting the frame of video into squares or rectangles in dimensions defined previously. 
* [resize_training_images.py](https://github.com/rejunges/Reconhecimento-de-Placas/blob/master/Scripts/resize_training_images.py)
	- Resizes the training images to the same dimension but the images below 35x35 are eliminated.
* [script_GT.py](https://github.com/rejunges/Reconhecimento-de-Placas/blob/master/Scripts/script_GT.py)
	- Reads the ground truth file(s) and remove the lines where the image dimension are inferior to 48x48.
* [create_digits.py](https://github.com/rejunges/Reconhecimento-de-Placas/blob/master/Scripts/create_digits.py)
	- Creates digits (0-9) and saves it in different folder to use in machine learning algorithms.
