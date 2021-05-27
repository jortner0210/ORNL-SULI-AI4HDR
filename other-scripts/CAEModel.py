import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import os

from tqdm import tqdm

from tensorflow.keras import optimizers
from pathlib import Path
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, InputSpec
from sklearn.cluster import KMeans


LATENT_DIMS = 2048


class HDRClusteringLayer(Layer):
	'''
	SOURCE: https://github.com/XifengGuo/DCEC/blob/master/DCEC.py

	This layer maps a sample's latent space vector to a vector that represents the probability
	of the sample belongining to each cluster.

	# Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
	'''

	def __init__(self, n_clusters=32, input_dim=LATENT_DIMS, alpha=1.0, **kwargs):
		super(HDRClusteringLayer, self).__init__()
		self.alpha 		= alpha
		self.n_clusters = n_clusters
		self.clusters   = self.add_weight(shape=(n_clusters, input_dim), initializer="glorot_uniform", trainable=True)
	

	def call(self, inputs):
		''' 
		student t-distribution, as same as used in t-SNE algorithm.
            q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        '''
		q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
		q **= (self.alpha + 1.0) / 2.0
		q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
		return q

	def get_config(self):
		config = {'n_clusters': self.n_clusters}
		base_config = super(HDRClusteringLayer, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))



class HDRClusterEncoder(keras.Model):

	def __init__(self, init_model=True, lr=0.001, latent_dim=LATENT_DIMS, n_clusters=10, activation="elu", stride=2, **kwargs):
		super(HDRClusterEncoder, self).__init__(**kwargs)

		self.opt = optimizers.Adam(learning_rate=lr)

		self.n_clusters = n_clusters
		self.y_pred = []

		self.cae     = None
		self.encoder = None
		self.model   = None

		if init_model:
			# Encoder/Decoder network
			self.cae = self.__buildCAE(activation, stride, latent_dim)
			self.cae.compile(optimizer=self.opt, loss="mse")

			# Pull out the embedded layer, and encoder for convenience
			hidden = self.cae.get_layer(name="latent_space").output
			self.encoder = keras.Model(inputs=self.cae.input, outputs=hidden, name="Encoder_Model")

			# Create Clustering Layer
			cluster_layer = HDRClusteringLayer(n_clusters=n_clusters, input_dim=latent_dim, name="clustering")(hidden)
			
			# Combined Encoder/Decoder/Cluster model
			self.model = keras.Model(inputs=self.cae.input, outputs=[cluster_layer, self.cae.output], name="DCEC_Model")


	@staticmethod
	def targetDistribution(q):
		weight = q ** 2 / q.sum(0)
		return (weight.T / weight.sum(1)).T


	def compile(self, loss=['kld', 'mse'], loss_weights=[1, 1]):
		self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=self.opt)


	def pretrain(self, x, batch_size=1, epochs=1, validation_split=0.0):
		# This method trains just the autoencoder.
		# Should be used before 'clusterTrain'
		return self.cae.fit(x, x, batch_size, epochs=epochs, verbose=1, validation_split=validation_split)


	def clusterTrain(self, data, batch_size, checkpoint_dir):
		tol = 0.001
		maxiter = 2e3 
		update_interval = 140

		print("Cluster train... Data shape=", data.shape, sep="")

		# Begin training by initializing the cluster layer with the clusters
		# learned from the sklearn KMeans algorithm
		kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
		self.y_pred = kmeans.fit_predict(self.encoder.predict(data))
		y_pred_last = np.copy(self.y_pred)
		self.model.get_layer(name='hdr_clustering_layer').set_weights([kmeans.cluster_centers_])

		all_loss = []

		loss = [0, 0, 0]
		index = 0
		for ite in tqdm(range(int(maxiter))):
			if ite % update_interval == 0:
				self.saveCheckPoint(checkpoint_dir)

				# Train model iteration
				q, cae_output = self.model.predict(data, verbose=0)
				p = self.targetDistribution(q)
 
				self.y_pred = q.argmax(1)

				# evaluate cluster performance, check stop criterion
				delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[0]
				y_pred_last = np.copy(self.y_pred)

				if ite > 0 and delta_label < tol:
					print('delta_label ', delta_label, '< tol ', tol)
					print('Reached tolerance threshold. Stopping training.')
					break

			# Train on batch.
			# y is made up of two components:
			# 	1) p - calculated with method targetDistribution
			#	   This is the target for the clustering layer
			#
			#	2) data - unaltered image data
			#	   This is the target for the decoder (reconstruction)

			if (index + 1) * batch_size > data.shape[0]:
				loss = self.model.train_on_batch(x=data[index * batch_size::], 
												 y=[p[index * batch_size::], data[index * batch_size::]])
				index = 0
			else:
				loss = self.model.train_on_batch(x=data[index * batch_size:(index + 1) * batch_size], 
												 y=[p[index * batch_size:(index + 1) * batch_size],
												 data[index * batch_size:(index + 1) * batch_size]])
				index += 1

			if loss != [0, 0, 0]:
				all_loss.append(loss)

		save_loss_file = "{}/LOSS-{}-{}-{}.csv".format(str(checkpoint_dir), 
													   self.model.metrics_names[0], 
													   self.model.metrics_names[1], 
													   self.model.metrics_names[2])

		np.savetxt(save_loss_file, np.array(all_loss), delimiter=",")

		self.saveCheckPoint(checkpoint_dir)
			

	def __buildCAE(self, activation="elu", stride=2, latent_dim=LATENT_DIMS):
		# ENCODER
		encoder_inputs = keras.Input(shape=(256, 256, 3))

		# Conv2D(filters, kernel_size, activation, strides, padding)
		layer_1 = layers.Conv2D(64,  5, activation=activation, strides=stride, padding="same")(encoder_inputs)	#16
		layer_2 = layers.Conv2D(64,  5, activation=activation, strides=stride, padding="same")(layer_1)#32
		layer_3 = layers.Conv2D(128, 5, activation=activation, strides=stride, padding="same")(layer_2)#64
		layer_4 = layers.Conv2D(256, 3, activation=activation, strides=stride, padding="same")(layer_3)#128
		layer_5 = layers.Flatten()(layer_4)
		latent  = layers.Dense(latent_dim, activation=activation, name="latent_space")(layer_5)

		# DECODER
		layer_5 = layers.Dense(16 * 16 * 256, activation=activation)(latent)
		layer_6 = layers.Reshape((16, 16, 256))(layer_5)

		layer_7 = layers.Conv2DTranspose(128, 5, activation=activation, strides=stride, padding="same")(layer_6)#64
		layer_8 = layers.Conv2DTranspose(64, 5, activation=activation, strides=stride, padding="same")(layer_7)#32
		layer_9 = layers.Conv2DTranspose(64, 5, activation=activation, strides=stride, padding="same")(layer_8)#16

		decoder_outputs = layers.Conv2DTranspose(3, 5, activation=activation, strides=stride, padding="same")(layer_9)

		cae = keras.Model(encoder_inputs, decoder_outputs, name="CAE")
		return cae


	'''
	Model save a load functions
	'''
	def saveCheckPoint(self, save_dir: str):
		save_dir = Path(save_dir)
		if not save_dir.is_dir():
			os.mkdir(save_dir)	

		self.saveModel(self.cae, "cae", save_dir)
		self.saveModel(self.encoder, "encoder", save_dir)
		self.saveModel(self.model, "model", save_dir)


	@staticmethod
	def loadCheckPoint(model_dir: str):

		hdr_model = HDRClusterEncoder(init_model=False)

		hdr_model.cae = hdr_model.loadModel("cae", model_dir)
		hdr_model.cae.compile(optimizer="adam", loss="mse")

		hdr_model.encoder = hdr_model.loadModel("encoder", model_dir)
		hdr_model.model   = hdr_model.loadModel("model", model_dir)

		hdr_model.n_clusters = hdr_model.model.get_layer(index=-2).output_shape[-1]

		return hdr_model


	@staticmethod
	def loadModel(model_type: str, save_dir: str):
		json_file = open("{}/{}.json".format(save_dir, model_type), 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json, custom_objects={"HDRClusteringLayer": HDRClusteringLayer})
		# load weights into new model
		loaded_model.load_weights("{}/{}.h5".format(save_dir, model_type))
		return loaded_model


	@staticmethod
	def saveModel(model: keras.Model, model_name: str, save_dir: str):
		model_json = model.to_json()
		with open("{}/{}.json".format(save_dir, model_name), "w") as json_file:
			json_file.write(model_json)
		model.save_weights("{}/{}.h5".format(save_dir, model_name))
