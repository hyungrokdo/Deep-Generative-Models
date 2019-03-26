# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 12:48:20 2018

@author: Hyungrok Do
         hyungrok.do11@gmail.com
         https://github.com/hyungrokdo
         
         A tensorflow-layer API implementation of AnoGAN
         
         Donahue, J., Krähenbühl, P., & Darrell, T. (2016).
         Schlegl, T., Seeböck, P., Waldstein, S. M., Schmidt-Erfurth, U., & Langs, G. (2017, June).
         Unsupervised anomaly detection with generative adversarial networks to guide marker discovery.
         In International Conference on Information Processing in Medical Imaging (pp. 146-157). Springer, Cham.
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=False, reshape=False)

def generator(z_in, use_batchnorm=True, use_bias=True):
    reuse = tf.AUTO_REUSE
    xavier_init, xavier_init_conv = tf.contrib.layers.xavier_initializer(uniform=True), tf.contrib.layers.xavier_initializer_conv2d(uniform=True)
    with tf.variable_scope('generator', reuse=reuse):
        net = tf.layers.dense(inputs=z_in, units=7*7*32, kernel_initializer=xavier_init, use_bias=use_bias, name='layer1/dense', reuse=reuse)
        net = tf.reshape(net, (-1, 7, 7, 32), name='layer1/reshape')
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=3, name='layer1/batchnorm', reuse=reuse)
        net = tf.nn.leaky_relu(net, name='layer1/act')
        
        net = tf.layers.conv2d_transpose(inputs=net, filters=16, kernel_size=(3, 3), strides=(2, 2), use_bias=use_bias, padding='same',
                                         kernel_initializer=xavier_init_conv, name='layer2/convtr', reuse=reuse)
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=3, name='layer2/batchnorm', reuse=reuse)
        net = tf.nn.leaky_relu(net, name='layer2/act')
        
        net = tf.layers.conv2d_transpose(inputs=net, filters=8, kernel_size=(3, 3), strides=(2, 2), use_bias=use_bias, padding='same',
                                         kernel_initializer=xavier_init_conv, name='layer3/convtr', reuse=reuse)
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=3, name='layer3/batchnorm', reuse=reuse)
        net = tf.nn.leaky_relu(net, name='layer3/act')
        
        net = tf.layers.conv2d_transpose(inputs=net, filters=1, kernel_size=(3, 3), strides=(1, 1), use_bias=use_bias, padding='same',
                                         kernel_initializer=xavier_init_conv, name='layer4/output', reuse=reuse)
    return tf.tanh(net)
        
def discriminator(x_in, use_batchnorm=False, use_bias=True):
    reuse = tf.AUTO_REUSE
    xavier_init_conv = tf.contrib.layers.xavier_initializer_conv2d(uniform=True)
    with tf.variable_scope('discriminator', reuse=reuse):
        net = tf.layers.conv2d(inputs=x_in, filters=16, kernel_size=(3, 3), strides=(2, 2), use_bias=use_bias, padding='same',
                               kernel_initializer=xavier_init_conv, name='layer1/conv', reuse=reuse)
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=3, name='layer1/batchnorm', reuse=reuse)
        net = tf.nn.leaky_relu(net, name='layer1/act')
        
        net = tf.layers.conv2d(inputs=net, filters=32, kernel_size=(3, 3), strides=(2, 2), use_bias=use_bias, padding='same',
                               kernel_initializer=xavier_init_conv, name='layer2/conv', reuse=reuse)
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=3, name='layer2/batchnorm', reuse=reuse)
        net = tf.nn.leaky_relu(net, name='layer2/act')
        
        net = tf.layers.flatten(net, name='layer3/act')
        net = tf.layers.dense(inputs=net, units=1, name='layer3/output')
    return net

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

z_dim          = 50
is_train       = tf.placeholder(tf.bool, name='is_train')
z              = tf.placeholder(dtype=tf.float32, shape=[None, z_dim], name='z')
x              = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
G              = generator(z)
D_real, D_fake = discriminator(x), discriminator(G)
d_loss_real    = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real))
d_loss_fake    = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake))
g_loss         = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))
d_loss         = tf.reduce_mean(d_loss_real) + tf.reduce_mean(d_loss_fake)
d_acc          = tf.reduce_mean(tf.cast(tf.equal(tf.concat([tf.ones_like(D_real, tf.int32), tf.zeros_like(D_fake, tf.int32)], 0),
                                                 tf.concat([tf.cast(tf.greater(D_real, 0.5), tf.int32), tf.cast(tf.greater(D_fake, 0.5), tf.int32)], 0)), tf.float32))
g_vars         = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
d_vars         = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
update_ops     = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
d_update_ops   = [var for var in update_ops if 'discriminator' in var.name]
g_update_ops   = [var for var in update_ops if 'generator' in var.name]

with tf.control_dependencies(d_update_ops):
    d_opt1 = tf.train.AdamOptimizer(learning_rate=1E-3, name='D-optimizer-1').minimize(loss=d_loss, var_list=d_vars)
    d_opt2 = tf.train.AdamOptimizer(learning_rate=1E-4, name='D-optimizer-2').minimize(loss=d_loss, var_list=d_vars)
    
with tf.control_dependencies(g_update_ops):
    g_opt1 = tf.train.AdamOptimizer(learning_rate=1E-3, name='G-optimizer-1').minimize(loss=g_loss, var_list=g_vars)
    g_opt2 = tf.train.AdamOptimizer(learning_rate=1E-4, name='G-optimizer-2').minimize(loss=g_loss, var_list=g_vars)
    
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

train_dat = mnist.train.images*2 - 1
n_train = len(train_dat)

max_epoch = 200
minibatch_size = 256

pbar = tqdm(range(max_epoch))

d_opt, g_opt = d_opt1, g_opt1
g_loss_traj, d_loss_traj = [], []
for epoch in pbar:
    train_idx = np.arange(n_train)
    np.random.shuffle(train_idx)
    train_batch = chunks(train_idx, minibatch_size)
    
    if epoch == 150:
        d_opt, g_opt = d_opt2, g_opt2
        
    g_loss_stack, d_loss_stack, d_acc_stack = [], [], []
    for batch_idx in train_batch:
        batch_x = train_dat[batch_idx]
        batch_z = np.random.uniform(-1, 1, size=[len(batch_idx), z_dim])
        D_loss, D_acc, _ = sess.run([d_loss, d_acc, d_opt], feed_dict={x: batch_x, z: batch_z, is_train: True})
        _         = sess.run(g_opt,           feed_dict={z: batch_z, is_train: True})
        G_loss, _ = sess.run([g_loss, g_opt], feed_dict={z: batch_z, is_train: True})
        
        g_loss_stack.append(G_loss)
        d_loss_stack.append(D_loss)
        d_acc_stack.append(D_acc)
        
    g_loss_traj.append(np.mean(g_loss_stack))
    d_loss_traj.append(np.mean(d_loss_stack))
    pbar.set_description('G-loss: {:.4f} | D-loss: {:.4f} | D-accuracy: {:.4f}'.format(np.mean(g_loss_stack), np.mean(d_loss_stack), np.mean(d_acc_stack)))
    

plt.plot(g_loss_traj); plt.plot(d_loss_traj); plt.show()

batch_z = np.random.uniform(-1, 1, size=[16, z_dim])
samples = sess.run(G, feed_dict={z: batch_z, is_train: False})

plt.figure(figsize=(10, 10))
for i, sample in enumerate(samples):
    plt.subplot(4, 4, i+1)
    plt.imshow(sample.reshape(28, 28), cmap='gray')
        
plt.show()

### AnoGAN - mapping new observations to the latent space
    
def get_discriminator_feature(x_in, use_batchnorm=False, use_bias=True):
    reuse = True
    xavier_init_conv = tf.contrib.layers.xavier_initializer_conv2d(uniform=True)
    with tf.variable_scope('discriminator', reuse=reuse):
        net = tf.layers.conv2d(inputs=x_in, filters=16, kernel_size=(3, 3), strides=(2, 2), use_bias=use_bias, padding='same',
                               kernel_initializer=xavier_init_conv, name='layer1/conv', reuse=reuse)
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=False, axis=3, name='layer1/batchnorm', reuse=reuse)
        net = tf.nn.leaky_relu(net, name='layer1/act')

    return net

target_x            = tf.placeholder(dtype=tf.float32, shape=[1, 28, 28, 1], name='target_x')
target_z            = tf.get_variable('anogan/target_z', shape=[1, z_dim], initializer=tf.random_uniform_initializer(-1, 1), trainable=True)
mapped_x            = generator(target_z)
target_d_feature    = get_discriminator_feature(target_x)
mapped_d_feature    = get_discriminator_feature(mapped_x)
lam                 = 0.7
anogan_var          = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='anogan')
residual_loss       = tf.reduce_mean(tf.abs(target_x - mapped_x), axis=[1, 2, 3])
discrimination_loss = tf.reduce_mean(tf.abs(target_d_feature - mapped_d_feature), axis=[1, 2, 3])
mapping_loss        = (1-lam)*residual_loss + lam*discrimination_loss
mapping_loss_opt1   = tf.train.AdamOptimizer(learning_rate=1E-1, name='mapping-optimizer-1').minimize(loss=mapping_loss, var_list=anogan_var)
mapping_loss_opt2   = tf.train.AdamOptimizer(learning_rate=1E-2, name='mapping-optimizer-2').minimize(loss=mapping_loss, var_list=anogan_var)

uninitialized_variables = [var for var in tf.global_variables() if not(sess.run(tf.is_variable_initialized(var)))]
sess.run(tf.variables_initializer(uninitialized_variables))

query_x = mnist.test.images[2].reshape(1, 28, 28, 1)
sess.run(tf.variables_initializer(anogan_var))
mapping_loss_traj = []
mapping_loss_opt = mapping_loss_opt1
for i in range(150):
    if i == 50:
        mapping_loss_opt = mapping_loss_opt2
    loss, _ = sess.run([mapping_loss, mapping_loss_opt], feed_dict={target_x: query_x, is_train: False})
    mapping_loss_traj.extend(loss)

anomaly_score = mapping_loss[-1]

### Comparison of Query Image and Mapped Image
    
generated_x = sess.run(generator(target_z), feed_dict={is_train: False})
plt.figure(figsize=(14, 4))
plt.subplot(1, 3, 1)
plt.imshow(generated_x.reshape(28, 28), cmap='gray')
plt.title('Mapped Image')
plt.subplot(1, 3, 2)
plt.imshow(query_x.reshape(28, 28), cmap='gray')
plt.title('Query Image')
plt.subplot(1, 3, 3)
plt.plot(mapping_loss_traj)
plt.title('Mapping loss per iteration')
plt.show()
plt.close()

