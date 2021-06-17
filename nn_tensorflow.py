import tensorflow as tf
import matplotlib.pyplot as plt
import numpy 
import pandas as pd
from sklearn.model_selection import train_test_split
class myCallback(tf.keras.callbacks.Callback): # Callback to stop training after particular loss
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('loss')<0.4):
            print('\nLoss is low so cancelling training!')
            self.model.stop_training = True
callbacks = myCallback()
mnist = pd.read_csv('fashion-mnist_train.csv') # loading the dataset
x = mnist.iloc[0:,1:]
y = mnist['label']
x = numpy.array(x)
y = numpy.array(y)
y = y.reshape(y.shape[0],1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3) #spliting the dataset
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
import matplotlib
for i in range(0,10):  # ploting the dataset
    data_digit=x_train[i]
    data_digit_image=data_digit.reshape(28,28)
    plt.imshow(data_digit_image,cmap=matplotlib.cm.binary,interpolation='nearest')
    plt.axis('off')
    plt.figure(i+1)
plt.show()
x_train = x_train/255.0
x_test = x_test/255.0
model = tf.keras.models.Sequential( # MODEL
    [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation=tf.nn.relu),
        tf.keras.layers.Dense(10,activation=tf.nn.softmax)
    ]
)
from tensorflow.keras.optimizers import Adam
model.compile(optimizer = 'Adam',
             loss = 'sparse_categorical_crossentropy')
model.fit(x_train,y_train,epochs=10,callbacks=[callbacks])  # training the model
model.evaluate(x_test,y_test) #testing the model