import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

#setting style
plt.style.use([ 'science', 'notebook', 'grid','bright'])

#setting a seed
np.random.seed(123)

#domain definition
X = np.arange( -1, 1, 0.01 )
X_m = X.reshape(-1, 1)

#target function definition
cos = np.cos(X * np.pi*2)
cos_m = cos.reshape(-1,1)

#source of randomness
normal = np.random.normal(0, 0.5, size=(200,1) )

#y data
y_m = cos_m + normal

#functions
def fit_smart_nn(X, Y):
    #architecture
    i = Input( shape = X[0].shape )
    init_1 = tf.keras.initializers.VarianceScaling(
        scale=1.0, mode="fan_in", distribution="untruncated_normal", seed=None
    )
    x = Dense(units=12,
              kernel_initializer = init_1,
              activation='tanh'
              )(i)
    init_2 = tf.keras.initializers.VarianceScaling(
        scale=1.0, mode="fan_in", distribution="untruncated_normal", seed=None
    )
    x = Dense(units=1,
              kernel_initializer=init_2
              )(x)

    #fitting
    model = Model(i,x)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()] )
    def schedule(epoch, lr):
      if epoch >= 200:
        return 0.001
      else:
        return 0.01
    scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)
    r = model.fit(X, Y, epochs=400, callbacks=[scheduler])

    # plt.figure(figsize=(14, 6))
    # plt.plot(r.history['loss'], label='loss')
    # plt.title('Loss Function - Smart NN')
    # plt.xlabel('Epochs')
    # plt.ylabel('Squared Loss')
    # plt.legend()
    # plt.show()

    return model


def fit_dumb_nn(X, Y):
    #architecture
    i = Input(shape=X[0].shape)
    x = Dense(units=12,
              activation='tanh'
              )(i)
    x = Dense(units=1,
              )(x)

    #fitting
    model = Model(i, x)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])

    def schedule(epoch, lr):
        if epoch >= 200:
            return 0.001
        else:
            return 0.01

    scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)
    r = model.fit(X, Y, epochs=400, callbacks=[scheduler])

    # plt.figure(figsize=(14, 6))
    # plt.plot(r.history['loss'], label='loss')
    # plt.title('Loss Function - Dumb NN')
    # plt.xlabel('Epochs')
    # plt.ylabel('Squared Loss')
    # plt.legend()
    # plt.show()

    return model

def main():
    smart_nn = fit_smart_nn(X_m, y_m)
    dumb_nn = fit_dumb_nn(X_m, y_m)

    y_smart_est = smart_nn.predict(X_m)
    y_dumb_est = dumb_nn.predict( X_m )

    plt.figure(figsize=(14, 6))
    plt.plot(X_m, y_smart_est, label='Smart NN')
    plt.plot(X_m, y_dumb_est, label='Dumb NN')
    plt.plot(X_m, cos_m, label='Cos')
    plt.scatter(X_m, y_m, label='Cos + Normal', s=10, color= (0.7, 0.1, 0.1) )
    plt.title( 'Neural Network Comparison' )
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()