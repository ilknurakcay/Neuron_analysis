import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
from data import *
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from keras.layers import Input, Cropping2D, concatenate, Concatenate, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, add, Dense
from keras.optimizers import *
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.applications import vgg16
import cv2

class myUnet(object):

	def __init__(self, img_rows = 128, img_cols = 128):

		self.img_rows = img_rows
		self.img_cols = img_cols

	def load_data(self):

		mydata = dataProcess(self.img_rows, self.img_cols)
		imgs_train, imgs_mask_train = mydata.load_train_data()
		imgs_test = mydata.load_test_data()
		return imgs_train, imgs_mask_train, imgs_test



	def get_unet(self):
    # inputs = Input((self.img_rows, self.img_cols, 3))
    # inputs = Input((16, 16, 512))
		inputTensor = Input(shape=(128, 128, 3))
		vgg_conv = vgg16.VGG16(input_tensor=inputTensor,
                           weights='imagenet',
                           include_top=False,
                           input_shape=(128, 128, 3),
                           classes=2)
    # vgg_conv.summary()
    # vgg_conv = vgg16.VGG16(input_tensor= imgs_train, weights='imagenet', include_top=False)
    # imgs_train = vgg_conv.predict(imgs_train)
		x1 = vgg_conv.get_layer('block1_conv2').output
		print('x1  shape:', x1.shape)
		x2 = vgg_conv.get_layer('block2_conv2').output
		print('x2  shape:', x2.shape)
		x3 = vgg_conv.get_layer('block3_conv3').output
		print('x3  shape:', x3.shape)
		x4 = vgg_conv.get_layer('block4_conv3').output
		print('x4  shape:', x4.shape)
		x5 = vgg_conv.get_layer('block5_pool').output
		print('x5  shape:', x5.shape)
		x6 = vgg_conv.get_layer('block5_conv3').output
		print('x6  shape:', x6.shape)
 		# input_shape = (1, self.img_rows, self.img_cols)
		data_format = 'channels_last'
		num_classes = 2

    # conv6 =  Flatten()(inputs)
		conv52 = Conv2D(2048, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x5)
		conv52 = Conv2D(2048, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv52)
		print('conv52  shape:', conv52.shape)


		up62 = Conv2D(1024, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv52))
		print('up62  shape:', up62.shape)
		merge62 = concatenate([x6, up62], axis=3)
		conv62 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge62)
		conv62 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv62)
    # conv62 = Dropout(0.2)(conv62)

		up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv62))
		print('up6  shape:', up6.shape)


		merge6 = concatenate([x4, up6], axis=3)
		conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
		conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    # conv6 = Dropout(0.2)(conv6)

		up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
		print('up7  shape:', up7.shape)
		merge7 = concatenate([x3, up7], axis=3)
		conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
		conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    # conv7 = Dropout(0.2)(conv7)

		up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
		print('up8  shape:', up8.shape)

		merge8 = concatenate([x2, up8], axis=3)
		conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
		conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    # conv8 = Dropout(0.2)(conv8)

		up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
		print('up9  shape:', up9.shape)

		merge9 = concatenate([x1, up9], axis=3)
		conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
		conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
		conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
		conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    # conv10 = add([conv10, inputs])


		model = Model(inputTensor, conv10)

		model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
		return model


	def train(self):

		print("loading data")
		imgs_train, imgs_mask_train, imgs_test = self.load_data()
		print("loading data done")
		model = self.get_unet()
		#model = load_model('/content/drive/MyDrive/unet-yeni/unet.hdf5')
		print("got unet")
		
		model_checkpoint = ModelCheckpoint('/content/drive/MyDrive/unet-yeni/unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
		print('Fitting model...')
		history=model.fit(imgs_train, imgs_mask_train, batch_size=2, epochs=100, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
		history.history.keys()
		plt.plot(history.history['loss'],'b--',label='loss')   
		plt.plot(history.history['val_loss'],label='validation loss') 
		plt.ylabel('loss')
		plt.xlabel('epoch') 
		plt.legend()
		plt.savefig('/content/drive/MyDrive/unet-yeni/graph.png')
		
		print('predict test data')
		imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
		np.save('/content/drive/MyDrive/unet-yeni/results/imgs_mask_test.npy', imgs_mask_test)

	def save_img(self):

		print("array to image")
		imgs = np.load('/content/drive/MyDrive/unet-yeni/results/imgs_mask_test.npy')
		for i in range(imgs.shape[0]):
			img = imgs[i]
			img = array_to_img(img)
			img.save("/content/drive/MyDrive/unet-yeni/results/%d.jpg"%(i))
    
	def test_npy_to_img(self):
		imgs2 = np.load('/content/drive/MyDrive/unet-yeni/data/npydata/imgs_test.npy')
		for i in range(imgs2.shape[0]):
			img = imgs2[i]
			img = array_to_img(img)
			img.save("/content/drive/MyDrive/unet-yeni/test_npy/%d.jpg"%(i))



if __name__ == '__main__':
	myunet = myUnet()
	myunet.train()
	myunet.save_img()
	myunet.test_npy_to_img()





