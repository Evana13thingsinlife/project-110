# To Capture Frame
import cv2

# To process image array
import numpy as np


# import the tensorflow modules and load the model
import tensorflow as tf

model = tf.keras.models.load_model('keras_model.h5')

# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:

	# Reading / Requesting a Frame from the Camera 
	status , frame = camera.read()

	# if we were sucessfully able to read the frame
	if status:

		# Flip the frame
		frame = cv2.flip(frame , 1)
		
		
		
		#resize the frame
		test_frame = cv2.resize(frame,(224,224))
		# expand the dimensions
		test_frame = np.array(frame, dtype=np.float32)
		test_frame = np.expand_dims(test_frame,axis=0)
                test_frame= test_frame/255.0
                predictions = model.predict(test_frame)
                rock = int(predictions[0][0]*100)
		paper = int(predictions[0][1]*100)
		scissor = int(predictions[0][2]*100)

		# printing percentage confidence
		print(f"Rock: {rock} %, Paper: {paper} %, Scissor: {scissor} %")

		cv2.imshow('feed' , frame)
        
        
          

        

		# waiting for 1ms
		code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if code == 32:
			break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()
