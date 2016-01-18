from keras.models import Sequential
from keras.layers.core import AutoEncoder,TimeDistributedDense
from keras.layers.recurrent import LSTM
import numpy as np

batch_size = 100
nb_epoch = 100

class AE:
	def __init__(self,X_train,input_dim=285):
		self.X_train = X_train[:, np.newaxis, :]
		print("Modified X_train: ", X_train.shape)
		self.input_dim=input_dim


	def build(self):
		self.ae = Sequential()
		self.ae = self.build_lstm_autoencoder(self.ae)
		self.ae.compile(loss='mean_squared_error', optimizer='adam')
        # Do NOT use validation data with return output_reconstruction=True
		np.random.seed(0)
		self.ae.fit(self.X_train, self.X_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=1)
		prefiler_Xtrain=self.ae.predict(self.X_train,verbose=1)
		print 'X_train: ',prefiler_Xtrain.shape

	def build_lstm_autoencoder(self,autoencoder):
        # The TimeDistributedDense isn't really necessary, however you need a lot of GPU memory to do 784x394-394x784
		autoencoder.add(TimeDistributedDense(input_dim=self.input_dim,
                                         output_dim=self.input_dim))

		autoencoder.add(AutoEncoder(encoder=LSTM(input_dim=self.input_dim,
                                             output_dim=50,
                                             activation='tanh',
                                             return_sequences=True),
                                decoder=LSTM(input_dim=50,
                                             output_dim=self.input_dim,
                                             activation='tanh',
                                             return_sequences=True),
                                output_reconstruction=False))
		return autoencoder

	def configure(self):
		return self.ae.get_config(verbose=True)

	def get_feature(self,X_test):
		X_test=X_test[:,np.newaxis,:]
		print("Modified X_test:",X_test.shape)
		pred_test = self.ae.predict(X_test, verbose=1)
		print("X_test: ", pred_test.shape)
		return pred_test


