#https://www.tensorflow.org/guide/core/mlp_core
#https://github.com/tensorflow/docs/blob/master/site/en/guide/core/mlp_core.ipynb
# Use seaborn for countplot
#!pip install -q seaborn

import matplotlib
from keras import datasets, layers, models
from matplotlib import pyplot as plt
import seaborn as sns
import tempfile
import os
# Preset Matplotlib figure sizes
matplotlib.rcParams['figure.figsize'] = [9, 6]

import tensorflow_datasets as tfds
import tensorflow as tf

print(tf.__version__)
# Set random seed for reproducible results
tf.random.set_seed(22)

train_data, val_data, test_data = tfds.load("mnist",
                                            split=['train[10000:]', 'train[0:10000]', 'test'],
                                            batch_size=128, as_supervised=True)

# x_viz, y_viz = tfds.load("mnist", split=['train[:1500]'], batch_size=-1, as_supervised=True)[0]
# x_viz = tf.squeeze(x_viz, axis=3)

# for i in range(9):
#     plt.subplot(3,3,1+i)
#     plt.axis('off')
#     plt.imshow(x_viz[i], cmap='gray')
#     plt.title(f"True Label: {y_viz[i]}")
#     plt.subplots_adjust(hspace=.5)

# sns.countplot(y_viz.numpy());
# plt.xlabel('Digits')
# plt.title("MNIST Digit Distribution");

def preprocess(x, y):
  # Reshaping the data
  x = tf.reshape(x, shape=[-1, 784])
  # Rescaling the data
  x = x/255
  return x, y

train_data, val_data = train_data.map(preprocess), val_data.map(preprocess)

# x = tf.linspace(-2, 2, 201)
# x = tf.cast(x, tf.float32)
# plt.plot(x, tf.nn.relu(x));
# plt.xlabel('x')
# plt.ylabel('ReLU(x)')
# plt.title('ReLU activation function');

# x = tf.linspace(-4, 4, 201)
# x = tf.cast(x, tf.float32)
# plt.plot(x, tf.nn.softmax(x, axis=0));
# plt.xlabel('x')
# plt.ylabel('Softmax(x)')
# plt.title('Softmax activation function');

def xavier_init(shape):
  # Computes the xavier initialization values for a weight matrix
  in_dim, out_dim = shape
  xavier_lim = tf.sqrt(6.)/tf.sqrt(tf.cast(in_dim + out_dim, tf.float32))
  weight_vals = tf.random.uniform(shape=(in_dim, out_dim),
                                  minval=-xavier_lim, maxval=xavier_lim, seed=22)
  return weight_vals

class DenseLayer(tf.Module):

  def __init__(self, out_dim, weight_init=xavier_init, activation=tf.identity):
    # Initialize the dimensions and activation functions
    self.out_dim = out_dim
    self.weight_init = weight_init
    self.activation = activation
    self.built = False

  def __call__(self, x):
    if not self.built:
      # Infer the input dimension based on first call
      self.in_dim = x.shape[1]
      # Initialize the weights and biases using Xavier scheme
      self.w = tf.Variable(xavier_init(shape=(self.in_dim, self.out_dim)))
      self.b = tf.Variable(tf.zeros(shape=(self.out_dim,)))
      self.built = True
    # Compute the forward pass
    z = tf.add(tf.matmul(x, self.w), self.b)
    return self.activation(z)

class MLP(tf.Module):
  def __init__(self, layers):
      self.layers = layers
  @tf.function
  def __call__(self, x, preds=False):
     # Execute the model's layers sequentially
      for layer in self.layers:
         x = layer(x)
      return x

#Forward Pass: ReLU(784 x 700) x ReLU(700 x 500) x Softmax(500 x 10)
# hidden_layer_1_size = 700
# hidden_layer_2_size = 500
# output_size = 10
# mlp_model = MLP([
#     DenseLayer(out_dim=hidden_layer_1_size, activation=tf.nn.relu),
#     DenseLayer(out_dim=hidden_layer_2_size, activation=tf.nn.relu),
#     DenseLayer(out_dim=output_size)])

def cross_entropy_loss(y_pred, y):
    # Compute cross entropy loss with a sparse operation
    sparse_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_pred)
    return tf.reduce_mean(sparse_ce)

def accuracy(y_pred, y):
  # Compute accuracy after extracting class predictions
  class_preds = tf.argmax(tf.nn.softmax(y_pred), axis=1)
  is_equal = tf.equal(y, class_preds)
  return tf.reduce_mean(tf.cast(is_equal, tf.float32))

class Adam:

    def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, ep=1e-7):
        # Initialize optimizer parameters and variable slots
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.learning_rate = learning_rate
        self.ep = ep
        self.t = 1.
        self.v_dvar, self.s_dvar = [], []
        self.built = False

    def apply_gradients(self, grads, vars):
        # Initialize variables on the first call
        if not self.built:
            for var in vars:
                v = tf.Variable(tf.zeros(shape=var.shape))
                s = tf.Variable(tf.zeros(shape=var.shape))
                self.v_dvar.append(v)
                self.s_dvar.append(s)
            self.built = True
        # Update the model variables given their gradients
        for i, (d_var, var) in enumerate(zip(grads, vars)):
            self.v_dvar[i].assign(self.beta_1 * self.v_dvar[i] + (1 - self.beta_1) * d_var)
            self.s_dvar[i].assign(self.beta_2 * self.s_dvar[i] + (1 - self.beta_2) * tf.square(d_var))
            v_dvar_bc = self.v_dvar[i] / (1 - (self.beta_1 ** self.t))
            s_dvar_bc = self.s_dvar[i] / (1 - (self.beta_2 ** self.t))
            var.assign_sub(self.learning_rate * (v_dvar_bc / (tf.sqrt(s_dvar_bc) + self.ep)))
        self.t += 1.
        return

def train_step(x_batch, y_batch, loss, acc, model, optimizer):
    # Update the model state given a batch of data
    with tf.GradientTape() as tape:
        y_pred = model(x_batch)
        batch_loss = loss(y_pred, y_batch)
    batch_acc = acc(y_pred, y_batch)
    grads = tape.gradient(batch_loss, model.variables)
    optimizer.apply_gradients(grads, model.variables)
    return batch_loss, batch_acc

def val_step(x_batch, y_batch, loss, acc, model):
    # Evaluate the model on given a batch of validation data
    y_pred = model(x_batch)
    batch_loss = loss(y_pred, y_batch)
    batch_acc = acc(y_pred, y_batch)
    return batch_loss, batch_acc

def train_model(mlp, train_data, val_data, loss, acc, optimizer, epochs):
  # Initialize data structures
  train_losses, train_accs = [], []
  val_losses, val_accs = [], []

  # Format training loop and begin training
  for epoch in range(epochs):
    batch_losses_train, batch_accs_train = [], []
    batch_losses_val, batch_accs_val = [], []

    # Iterate over the training data
    for x_batch, y_batch in train_data:
      # Compute gradients and update the model's parameters
      batch_loss, batch_acc = train_step(x_batch, y_batch, loss, acc, mlp, optimizer)
      # Keep track of batch-level training performance
      batch_losses_train.append(batch_loss)
      batch_accs_train.append(batch_acc)

    # Iterate over the validation data
    for x_batch, y_batch in val_data:
      batch_loss, batch_acc = val_step(x_batch, y_batch, loss, acc, mlp)
      batch_losses_val.append(batch_loss)
      batch_accs_val.append(batch_acc)

    # Keep track of epoch-level model performance
    train_loss, train_acc = tf.reduce_mean(batch_losses_train), tf.reduce_mean(batch_accs_train)
    val_loss, val_acc = tf.reduce_mean(batch_losses_val), tf.reduce_mean(batch_accs_val)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    print(f"Epoch: {epoch}")
    print(f"Training loss: {train_loss:.3f}, Training accuracy: {train_acc:.3f}")
    print(f"Validation loss: {val_loss:.3f}, Validation accuracy: {val_acc:.3f}")
  return train_losses, train_accs, val_losses, val_accs

##########################################################################################################

hidden_layer_size = 64
output_size = 10
epoch_num=1
mlp_model1 = MLP([DenseLayer(out_dim=output_size)])
mlp_model2 = MLP([DenseLayer(out_dim=hidden_layer_size, activation=tf.nn.relu),
                  DenseLayer(out_dim=output_size)])

# train_losses1, train_accs1, val_losses1, val_accs1 = train_model(mlp_model1, train_data, val_data,
#                                                                  loss=cross_entropy_loss, acc=accuracy,
#                                                                  optimizer=Adam(), epochs=epoch_num)
#
# train_losses2, train_accs2, val_losses2, val_accs2 = train_model(mlp_model2, train_data, val_data,
#                                                                   loss=cross_entropy_loss, acc=accuracy,
#                                                                   optimizer=Adam(), epochs=epoch_num)

########################################################################################################

train_images, train_labels = tfds.load("mnist", split=['train[10000:]'], batch_size=-1, as_supervised=True)[0]
# train_images = tf.squeeze(train_images, axis=3)
train_images = tf.cast(train_images, tf.float32) / 255.0

val_images, val_labels = tfds.load("mnist", split=['train[0:10000]'], batch_size=-1, as_supervised=True)[0]
val_images = tf.cast(val_images, tf.float32) / 255.0

epoch_num=8

model3 = tf.keras.Sequential()
model3.add(layers.Conv2D(64, (3, 3), input_shape=(28, 28, 1)))
model3.add(layers.MaxPooling2D((2, 2)))
model3.add(layers.Flatten())
model3.add(layers.Dense(10))

model4 = tf.keras.Sequential()
model4.add(layers.Conv2D(64, (3, 3), input_shape=(28, 28, 1)))
model4.add(layers.MaxPooling2D((2, 2)))
model4.add(layers.Flatten())
model4.add(layers.Dense(64, activation=tf.nn.relu))
model4.add(layers.Dense(10))

model3.compile(optimizer=tf.keras.optimizers.Adam(),
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

history3 = model3.fit(train_images, train_labels, epochs=epoch_num,
                      validation_data=(val_images, val_labels))

model4.compile(optimizer=tf.keras.optimizers.Adam(),
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

history4 = model4.fit(train_images, train_labels, epochs=epoch_num,
                      validation_data=(val_images, val_labels))

# train_losses3 = history3.history['loss']
# train_accs3 = history3.history['accuracy']
# val_losses3 = history3.history['val_loss']
# val_accs3 = history3.history['val_accuracy']
#
# train_losses4 = history4.history['loss']
# train_accs4 = history4.history['accuracy']
# val_losses4 = history4.history['val_loss']
# val_accs4 = history4.history['val_accuracy']

############################################################################################################

def plot_metrics(model1_metric, model2_metric, model3_metric, model4_metric, metric_type):
  # Visualize metrics vs training Epochs
  plt.figure()
  plt.plot(range(len(model1_metric)), model1_metric, label = f"Simple model with Softmax {metric_type}")
  plt.plot(range(len(model2_metric)), model2_metric, label = f"Model with Relu and Softmax {metric_type}")
  plt.plot(range(len(model3_metric)), model3_metric, label=f"CNN with Softmax {metric_type}")
  plt.plot(range(len(model4_metric)), model4_metric, label=f"CNN with Relu and Softmax {metric_type}")
  plt.xlabel("Epochs")
  plt.ylabel(metric_type)
  plt.legend()
  plt.title(f"{metric_type} vs epochs");
  plt.show()

# plot_metrics(train_losses1, train_losses2, train_losses3, train_losses4, "training loss")
# plot_metrics(train_accs1, train_accs2, train_accs3, train_accs4, "training accuracy")
# plot_metrics(val_losses1, val_losses2, val_losses3, val_losses4, "validation loss")
# plot_metrics(val_accs1, val_accs2, val_accs3, val_accs4, "validation accuracy")

###########################################################################################################
import numpy as ny

class ExportModule(tf.Module):
  def __init__(self, model, preprocess, class_pred):
    # Initialize pre and postprocessing functions
    self.model = model
    self.preprocess = preprocess
    self.class_pred = class_pred

  @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, None], dtype=tf.uint8)])
  def __call__(self, x):
    # Run the ExportModule for new data points
    x = self.preprocess(x)
    y = self.model(x)
    y = self.class_pred(y)
    return y

def preprocess_test(x):
  # The export module takes in unprocessed and unlabeled data
  x = tf.reshape(x, shape=[-1, 784])
  x = x/255
  return x

def class_pred_test(y):
  # Generate class predictions from MLP output
  return tf.argmax(tf.nn.softmax(y), axis=1)

def accuracy_score(y_pred, y):
  # Generic accuracy function
  is_equal = tf.equal(y_pred, y)
  return tf.reduce_mean(tf.cast(is_equal, tf.float32))

epoch_num=1
mlp_model=mlp_model1

train_losses, train_accs, val_losses, val_accs = train_model(mlp_model, train_data, val_data,
                                                             loss=cross_entropy_loss, acc=accuracy,
                                                             optimizer=Adam(), epochs=epoch_num)

mlp_model_export = ExportModule(model=mlp_model,
                                preprocess=preprocess_test,
                                class_pred=class_pred_test)

models = tempfile.mkdtemp()
save_path = os.path.join(models, 'mlp_model_export')
tf.saved_model.save(mlp_model_export, save_path)
mlp_loaded = tf.saved_model.load(save_path)

test_images, test_labels = tfds.load("mnist", split=['test'], batch_size=-1, as_supervised=True)[0]

test_classes = mlp_loaded(test_images)

test_acc = accuracy_score(test_classes, test_labels)
print(f"Test Accuracy: {test_acc:.3f}")

print("Accuracy breakdown by digit:")
print("---------------------------")
label_accs = {}
for label in range(10):
  label_ind = (test_labels == label)
  # extract predictions for specific true label
  pred_label = test_classes[label_ind]
  label_filled = tf.cast(tf.fill(pred_label.shape[0], label), tf.int64)
  # compute class-wise accuracy
  label_accs[accuracy_score(pred_label, label_filled).numpy()] = label
for key in sorted(label_accs):
  print(f"Digit {label_accs[key]}: {key:.3f}")

import sklearn.metrics as sk_metrics

def show_confusion_matrix(test_labels, test_classes):
  # Compute confusion matrix and normalize
  plt.figure(figsize=(10,10))
  confusion = sk_metrics.confusion_matrix(test_labels.numpy(),
                                          test_classes.numpy())
  confusion_normalized = confusion / confusion.sum(axis=1)
  axis_labels = range(10)
  ax = sns.heatmap(
      confusion_normalized, xticklabels=axis_labels, yticklabels=axis_labels,
      cmap='Blues', annot=True, fmt='.4f', square=True)
  plt.title("Confusion matrix")
  plt.ylabel("True label")
  plt.xlabel("Predicted label")

# show_confusion_matrix(test_labels, test_classes)
# plt.show()

# ##########################################################################################

# # plt.plot(history.history['accuracy'], label='accuracy')
# # plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# # plt.xlabel('Epoch')
# # plt.ylabel('Accuracy')
# # plt.ylim([0.5, 1])
# # plt.legend(loc='lower right')

epoch_num=8
model=model4
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=epoch_num,
                    validation_data=(val_images, val_labels))

def class_pred_test(y):
  # Generate class predictions from MLP output
    return tf.argmax(tf.nn.softmax(y), axis=1)

test_images1, test_labels = tfds.load("mnist", split=['test'], batch_size=-1, as_supervised=True)[0]

test_images1 = tf.cast(test_images1, tf.float32) / 255.0

y=model(test_images1)
test_classes1 = class_pred_test(y)
test_acc = accuracy_score(test_classes1, test_labels)
print(f"Test Accuracy: {test_acc:.3f}")
# test_loss, test_acc = model.evaluate(test_images1,  test_labels, verbose=True)
# print(test_loss, test_acc)

print("Accuracy breakdown by digit:")
print("---------------------------")
label_accs = {}
for label in range(10):
  label_ind = (test_labels == label)
  # extract predictions for specific true label
  pred_label = test_classes[label_ind]
  label_filled = tf.cast(tf.fill(pred_label.shape[0], label), tf.int64)
  # compute class-wise accuracy
  label_accs[accuracy_score(pred_label, label_filled).numpy()] = label
for key in sorted(label_accs):
  print(f"Digit {label_accs[key]}: {key:.3f}")

def show_confusion_matrix(test_labels, test_classes):
  # Compute confusion matrix and normalize
  plt.figure(figsize=(10,10))
  confusion = sk_metrics.confusion_matrix(test_labels.numpy(),
                                          test_classes.numpy())
  confusion_normalized = confusion / confusion.sum(axis=1)
  axis_labels = range(10)
  ax = sns.heatmap(
      confusion_normalized, xticklabels=axis_labels, yticklabels=axis_labels,
      cmap='Blues', annot=True, fmt='.4f', square=True)
  plt.title("Confusion matrix")
  plt.ylabel("True label")
  plt.xlabel("Predicted label")

show_confusion_matrix(test_labels, test_classes)
plt.show()

# #############################################################################################

from contextlib import redirect_stdout

def predictwrongs(pred_labels, test_labels,  filename):
    result = []
    with open(filename, 'w') as f:
        with redirect_stdout(f):
            j = 0
            for i in range(10000):
                 pred_label = pred_labels[i]
                 test_label = test_labels[i]
                 if tf.cast(pred_label, tf.float32)!=tf.cast(test_label, tf.float32):
                    result.append(i)
                    j+=1

                    #print("Image " + str(i) + " expected as " + str(test_label) + " is wrongfully predicted as " + str(pred_label)
                    print("(" + str(i) + ", " + str(test_label) + ", " + str(pred_label) + ")")
            print("In total of {} wrong predictions for this testing data".format(j))
    f.close()
    return result

# pred_labels = mlp_loaded(test_images)
# pred_labels = class_pred_test(y)
# test_labels = test_labels

# filename = "wrongs1.txt"
# filename = "wrongs2.txt"
# filename = "wrongs3.txt"
# filename = "wrongs4.txt"
# print(predictwrongs(pred_labels, test_labels,  filename))

#
test_images, test_labels = tfds.load("mnist", split=['test'], batch_size=-1, as_supervised=True)[0]
test_images = tf.squeeze(test_images, axis=3)
z = [46, 84, 88, 104, 129]
for i in range(5):
    plt.subplot(1,5,1+i)
    plt.axis('off')
    plt.imshow(test_images[z[i]], cmap='gray')
    plt.title(f"{test_labels[z[i]]}")
    plt.subplots_adjust(hspace=.5)
plt.show()

z = [103, 369, 1469, 1532, 1945]
for i in range(5):
    plt.subplot(1,5,1+i)
    plt.axis('off')
    plt.imshow(test_images[z[i]], cmap='gray')
    plt.title(f"{test_labels[z[i]]}")
    plt.subplots_adjust(hspace=.5)
plt.show()

z = [164, 639, 720, 1546, 1898]
for i in range(5):
    plt.subplot(1,5,1+i)
    plt.axis('off')
    plt.imshow(test_images[z[i]], cmap='gray')
    plt.title(f"{test_labels[z[i]]}")
    plt.subplots_adjust(hspace=.5)
plt.show()

z = [942, 1381, 3025, 3063, 3315]
for i in range(5):
    plt.subplot(1,5,1+i)
    plt.axis('off')
    plt.imshow(test_images[z[i]], cmap='gray')
    plt.title(f"{test_labels[z[i]]}")
    plt.subplots_adjust(hspace=.5)
plt.show()