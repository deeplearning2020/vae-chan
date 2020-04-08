from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K

class latent_vector(Layer):
    def __init__(self, latent_regularizer = 'bvae', beta = 100., **kwargs):
        super(latent_vector, self).__init__(**kwargs)

    def build(self, input_shape):
        super(latent_vector, self).build(input_shape)

    def call(self, x, training = None):
        if len(x) != 2:
            raise Exception('input layers must be a list: mean and logvar')
        if len(x[0].shape) != 2 or len(x[1].shape) != 2:
            raise Exception('input shape is not a vector [batchSize, latentSize]')
        mean = x[0]
        logvar = x[1]
        if mean.shape[0].value == None or  logvar.shape[0].value == None:
            return mean + 0*logvar

        if self.reg is not None:
            latent_loss = -0.5 * (1 + logvar
                                - K.square(mean)
                                - K.exp(logvar))
            latent_loss = K.sum(latent_loss, axis = -1)
            latent_loss = K.mean(latent_loss, axis = 0)
            latent_loss = self.beta * latent_loss
            self.add_loss(latent_loss, x)

        def reparameterization_trick():
            epsilon = K.random_normal(shape = logvar.shape,
                              mean= 0., stddev = 1.)
            stddev = K.exp(logvar*0.5)
            return mean + stddev * epsilon

        return K.in_train_phase(reparameterization_trick, mean + 0*logvar, training = training)

    def compute_output_shape(self, input_shape):
        return input_shape[0]
