import keras
from keras import layers
from keras import activations
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Normalization
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

def polar_generator(batchsize,grid=(10,10),noise=.002,flat=False):
  while True:
    x = np.random.rand(batchsize)
    y = np.random.rand(batchsize)
    out = np.zeros((batchsize,grid[0],grid[1]))
    xc = (x*grid[0]).astype(int)
    yc = (y*grid[1]).astype(int)
    for b in range(batchsize):
      out[b,xc[b],yc[b]] = 1
    #compute rho and theta and add some noise
    rho = np.sqrt(x**2+y**2) + np.random.normal(scale=noise)
    theta = np.arctan(y/np.maximum(x,.00001)) + np.random.normal(scale=noise)
    if flat:
      out = np.reshape(out,(batchsize,grid[0]*grid[1]))
    yield ((theta,rho),out)

n_train = 4000000
n_test = 20000
batch_size = 4096

g1,g2 = 10,10
gen = polar_generator(n_train+n_test,grid=(g1,g2),noise=0.002,flat=True)
# (theta,rho),y = next(gen)
(theta,rho),y = next(gen)
x=np.array([i for i in zip(theta,rho)])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=n_test/(n_train+n_test), shuffle=True, random_state=1)

def discretized_accuracy(true_maps: tf.Tensor, my_maps: tf.Tensor) -> float:
  equals = tf.equal(tf.argmax(true_maps, axis=1), tf.argmax(my_maps, axis=1))
  return tf.cast(tf.math.count_nonzero(equals), tf.float64) / tf.cast(len(true_maps), tf.float64)

theta_input = Input(shape=(1,), name="theta")
theta_norm = Normalization(axis=None)
theta_norm.adapt(x_train[:,0])
theta_norm = theta_norm(theta_input)
theta_branch = Dense(2, activation=activations.softsign)(theta_norm)
theta_branch = Dense(4, activation=activations.tanh)(theta_branch)
theta_branch = Dense(4, activation=activations.sigmoid)(theta_branch)

rho_input = Input(shape=(1,), name="rho")
rho_norm = Normalization(axis=None)
rho_norm.adapt(x_train[:,1])
rho_norm = rho_norm(rho_input)
# rho_branch = rho_norm
rho_branch = Dense(4, activation=activations.softsign)(rho_norm)
rho_branch = Dense(4, activation=activations.swish)(rho_branch)
rho_branch = Dense(4, activation=activations.tanh)(rho_branch)
rho_branch = Dense(4, activation=activations.elu)(rho_branch)

concatenate_layer = Concatenate(name="concatenation")([theta_branch, rho_branch])
output = layers.Dense(8, activation=keras.activations.swish)(concatenate_layer)
output = layers.Dense(16, activation=keras.activations.relu)(output)
output = layers.Dense(4, activation=keras.activations.gelu)(output)
output = layers.Dense(100, activation=activations.softmax)(output)

network = Model([theta_input, rho_input], output)

# Prima di poter usare il modello dobbiamo dire a Keras la dimensione dei nostri input
# "None" vuol dire che il numero è ignoto/può cambiare (perché quante immagini alla volta g# può cambiare)
network.build((None, 2))
network.summary()

network.compile(
optimizer=keras.optimizers.Adam(learning_rate=1e-3),
loss=keras.losses.CategoricalCrossentropy(),
# loss=customLoss,
metrics=['accuracy', discretized_accuracy]
)

load = input('wanna load previous weights?')
if load == 'y':
    network = keras.models.load_model("prova_luca")

while True:
    history = network.fit(
      x=[x_train[:,0], x_train[:,1]],
      y=y_train,
      epochs=15, # Addestriamo per 100 epoche
      batch_size=batch_size, # Usiamo una batch size di 128
      validation_data=([x_test[:,0], x_test[:,1]], y_test),
      callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
    )
    train = input('train again?')
    if train != 'y':
        break

score, _, acc  = network.evaluate([x_test[:,0], x_test[:,1]], y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Accuracy: {:.1f}%'.format(acc*100))

save = input('wanna load previous weights?')
if save == 'y':
    network.save("prova_luca")

while True:
    gen = polar_generator(20000,grid=(g1,g2),noise=0.002,flat=True)
    accs = 0.0
    lower = 0
    iters = 100
    for x in range(iters):
      (theta,rho),y = next(gen)
      x=np.array([i for i in zip(theta,rho)])

      score, _, acc = network.evaluate([x[:,0], x[:,1]], y, batch_size=batch_size)
      accs += acc
      if acc < 0.95:
        lower += 1

    print('------------------------')
    print('Accuracy: {:.1f}%'.format(accs/iters*100))
    print('Lower than 95%: {}/{}'.format(lower, iters))
    print('------------------------')

    test = input('test again?')
    if test != 'y':
        break
