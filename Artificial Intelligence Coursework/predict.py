import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse


# passes the image path, grabs the file name from the path and sets the variables and array
dir_path = os.path.dirname(os.path.realpath(__file__))
image_path=sys.argv[1] 
filename = dir_path +'/' +image_path
image_size=128
num_channels=3
images = []

# reads the image using opencv
image = cv2.imread(filename)

# uses opencv and numpPy the same as in the dataset.py to resize the image
image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0/155.0) 

# reshape the image
x_batch = images.reshape(1, image_size,image_size,num_channels)

# restore the saved model 
sess = tf.Session()
# recreate the network graph
saver = tf.train.import_meta_graph('emotions_model.meta')
# load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./'))

# access the restored graph
graph = tf.get_default_graph()


#  y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

# feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0") 
y_test_images = np.zeros((1, 7)) 


# create the feed_dict that is required to be fed to calculate y_pred 
feed_dict_testing = {x: x_batch, y_true: y_test_images}
result=sess.run(y_pred, feed_dict=feed_dict_testing)
# sets the emotions into integer values and outputs percentage prediction to the console
Angry = result[0][0]
Disgusted = result[0][1]
Fear = result[0][2]
Happy = result[0][3]
Neutral = result[0][4]
Sad = result[0][5]
Surprised = result[0][6]

int (Angry)
int (Disgusted)
int (Fear)
int (Happy)
int (Neutral)
int (Sad)
int (Surprised)
Angry = Angry * 100
Disgusted = Disgusted * 100
Fear = Fear * 100
Happy = Happy * 100
Neutral = Neutral * 100
Sad = Sad * 100
Surprised = Surprised * 100

print ('\n Angry: ', ("%.2f" % Angry), '% \n', 'Disgusted: ', ("%.2f" % Disgusted), '% \n', 'Fear: ', ("%.2f" % Fear), '% \n', 'Happy: ', ("%.2f" % Happy), '% \n', 'Neutral: ', ("%.2f" % Neutral), '% \n', 'Sad: ', ("%.2f" % Sad), '% \n', 'Surprised: ', ("%.2f" % Surprised), '% \n')

# prints out the outcome the network determined was the correct prediction

if Happy > Angry and Happy > Disgusted and Happy > Fear and Happy > Neutral and Happy > Sad and Happy > Surprised:
	print ("The person is Happy")
	
if Angry > Happy and Angry > Disgusted and Angry > Fear and Angry > Neutral and Angry > Sad and Angry > Surprised:
	print ("The person is Angry")

if Disgusted > Angry and Disgusted > Happy and Disgusted > Fear and Disgusted > Neutral and Disgusted > Sad and Disgusted > Surprised:
	print ("The person is Disgusted")

if Fear > Angry and Fear > Disgusted and Fear > Happy and Fear > Neutral and Fear > Sad and Fear > Surprised:
	print ("The person is Fearful")

if Neutral > Angry and Neutral > Disgusted and Neutral > Fear and Neutral > Happy and Neutral > Sad and Neutral > Surprised:
	print ("The person is Neutral")

if Sad > Angry and Sad > Disgusted and Sad > Fear and Sad > Neutral and Sad > Happy and Sad > Surprised:
	print ("The person is Sad")

if Surprised > Angry and Surprised > Disgusted and Surprised > Fear and Surprised > Neutral and Surprised > Sad and Surprised > Happy:
	print ("The person is Surprised")	