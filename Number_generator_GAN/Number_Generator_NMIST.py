import numpy as np
import pandas as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train/255
x_test = x_test/255
x_train = x_train.reshape(-1,28,28,1) * 2.0 - 1
x_train.min()
only_zeros = x_train[y_train==0]  
#variables having only 0 number images
# we can choose any numbers for the above line

#discreminator it uses binary classification

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Reshape,Flatten
from tensorflow.keras.layers import BatchNormalization,Flatten,MaxPooling2D
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,LeakyReLU,Dropout
from tensorflow.keras.layers import ELU
import tensorflow as tf

# =============================================================================
# Phase 2 model Generator
# =============================================================================
coding_size = 100
# 100 -> 150 -> 784
generator = Sequential()
generator.add(Dense(7*7*128, input_shape=[coding_size]))
generator.add(Reshape([7,7,128]))
generator.add(BatchNormalization())
generator.add(Conv2DTranspose(64,kernel_size=5,strides=2,padding='same',
                              activation='relu'))
generator.add(BatchNormalization())
generator.add(Conv2DTranspose(1,kernel_size=5,strides=2,padding='same',
                              activation='tanh'))
# =============================================================================
# Phase 1 model Discriminator
# =============================================================================
# 784 -> 150 -> 100 -> 1
discriminator = Sequential()
discriminator.add(Conv2D(64,kernel_size=5,strides=2,padding='same',
                         activation=LeakyReLU(0.3),
                         input_shape=[28,28,1]))
discriminator.add(Dropout(0.5))
discriminator.add(Conv2D(128,kernel_size=5,strides=2,padding='same',
                         activation=LeakyReLU(0.3),
                         input_shape=[28,28,1]))
discriminator.add(Dropout(0.5))
discriminator.add(Flatten())
discriminator.add(Dense(1,activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy',optimizer='adam')
discriminator.trainable = False

# =============================================================================
# Creaing the GAN model
# =============================================================================
GAN = Sequential([generator, discriminator])
GAN.compile(loss='binary_crossentropy',optimizer='adam')
GAN.summary()
GAN.layers[0].summary()
GAN.layers[1].summary()


# =============================================================================
# Setting up the traning batch
# =============================================================================
batch_size = 32
my_data = only_zeros
dataset = tf.data.Dataset.from_tensor_slices(my_data).shuffle(buffer_size=1000)
type(dataset)
dataset = dataset.batch(batch_size,drop_remainder=True).prefetch(1)
epochs=20


# =============================================================================
# Training Loop 
# =============================================================================
generator,discriminator = GAN.layers
for epoch in range(epochs):
    print(f'Currently on epoch: ', epoch + 1)
   
    i = 0
    for x_batch in dataset :
        i = i + 1
    
        if i%100 == 0:
            print(f'\tCurrently on batch number {i} of {len(my_data)//batch_size}' )
        
        #Descriminator training phase
        noise = tf.random.normal(shape=[batch_size,coding_size])
       
        gen_image = generator(noise)
        
        x_fake_vs_real = tf.concat([gen_image,tf.dtypes.cast(x_batch,tf.float32)],axis=0)
        
        y1 = tf.constant([[0.0]]*batch_size + [[1.0]]*batch_size)    
        
        discriminator.trainable = True
        
        discriminator.train_on_batch(x_fake_vs_real,y1)
        
        
        # Generator traning phase
        noise = tf.random.normal(shape=[batch_size,coding_size])
        
        y2 =tf.constant([[1.0]]*batch_size)
        
        discriminator.trainable = False
        
        GAN.train_on_batch(noise,y2)
        
        
        
        

noise =tf.random.normal(shape=[10,coding_size])
noise.shape
plt.imshow(noise)


images = generator(noise)
images.shape

for image in images:
    plt.imshow(image.numpy().reshape(28,28))
    plt.show()
#all the generated image are very similar and this is a problem
# this problem in known as Mode Collapse
# it a very common probelm 










