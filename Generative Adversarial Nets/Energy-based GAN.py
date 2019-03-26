# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:35:48 2019

@author: Hyungrok Do
         hyungrok.do11@gmail.com
         https://github.com/hyungrokdo
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=False, reshape=False)

def generator(z_in, use_batchnorm=True, use_bias=True):
    reuse = tf.AUTO_REUSE
    xavier_init, xavier_init_conv = tf.contrib.layers.xavier_initializer(uniform=True), tf.contrib.layers.xavier_initializer_conv2d(uniform=True)
    with tf.variable_scope('generator', reuse=reuse):
        net = tf.layers.dense(inputs=z_in, units=7*7*64, kernel_initializer=xavier_init, use_bias=use_bias, name='layer1/dense', reuse=reuse)
        net = tf.reshape(net, (-1, 7, 7, 64), name='layer1/reshape')
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=-1, name='layer1/batchnorm', reuse=reuse)
        net = tf.nn.leaky_relu(net, name='layer1/act')
        
        net = tf.layers.conv2d_transpose(inputs=net, filters=32, kernel_size=(5, 5), strides=(2, 2), use_bias=use_bias, padding='same',
                                         kernel_initializer=xavier_init_conv, name='layer2/convtr', reuse=reuse)
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=-1, name='layer2/batchnorm', reuse=reuse)
        net = tf.nn.leaky_relu(net, name='layer2/act')
        
        net = tf.layers.conv2d_transpose(inputs=net, filters=16, kernel_size=(5, 5), strides=(2, 2), use_bias=use_bias, padding='same',
                                         kernel_initializer=xavier_init_conv, name='layer3/convtr', reuse=reuse)
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=-1, name='layer3/batchnorm', reuse=reuse)
        net = tf.nn.leaky_relu(net, name='layer3/act')
        
        net = tf.layers.conv2d_transpose(inputs=net, filters=1, kernel_size=(5, 5), strides=(1, 1), use_bias=use_bias, padding='same',
                                         kernel_initializer=xavier_init_conv, name='layer4/output', reuse=reuse)
    return tf.nn.sigmoid(net)

def autoencoder(x_in, use_batchnorm=True, use_bias=True):
    ''' autoencoder as energy function/discriminator '''
    reuse = tf.AUTO_REUSE
    xavier_init_conv = tf.contrib.layers.xavier_initializer_conv2d(uniform=True)
    with tf.variable_scope('discriminator', reuse=reuse):
        net = tf.layers.conv2d(inputs=x_in, filters=8, kernel_size=(5, 5), strides=(1, 1), use_bias=use_bias, padding='same',
                               kernel_initializer=xavier_init_conv, name='layer1/conv', reuse=reuse)
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=-1, name='layer1/batchnorm', reuse=reuse)
        net = tf.nn.leaky_relu(net, alpha=0.15, name='layer1/act')
        
        net = tf.layers.conv2d(inputs=net, filters=16, kernel_size=(5, 5), strides=(2, 2), use_bias=use_bias, padding='same',
                               kernel_initializer=xavier_init_conv, name='layer2/conv', reuse=reuse)
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=-1, name='layer2/batchnorm', reuse=reuse)
        net = tf.nn.leaky_relu(net, alpha=0.15, name='layer2/act')
        
        net = tf.layers.conv2d(inputs=net, filters=32, kernel_size=(5, 5), strides=(2, 2), use_bias=use_bias, padding='same',
                               kernel_initializer=xavier_init_conv, name='layer3/conv', reuse=reuse)
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=-1, name='layer3/batchnorm', reuse=reuse)
        net = tf.nn.leaky_relu(net, alpha=0.15, name='layer3/act')
        
        net = tf.layers.conv2d(inputs=net, filters=32, kernel_size=(7, 7), strides=(1, 1), use_bias=use_bias, padding='valid',
                               kernel_initializer=xavier_init_conv, name='layer4/conv', reuse=reuse)
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=-1, name='layer4/batchnorm', reuse=reuse)
        net = tf.nn.leaky_relu(net, alpha=0.15, name='layer4/act')
        
        net = tf.layers.conv2d_transpose(inputs=net, filters=32, kernel_size=(7, 7), strides=(1, 1), use_bias=use_bias, padding='valid',
                                         kernel_initializer=xavier_init_conv, name='layer5/trconv', reuse=reuse)
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=-1, name='layer5/batchnorm', reuse=reuse)
        net = tf.nn.leaky_relu(net, alpha=0.15, name='layer5/act')
        
        net = tf.layers.conv2d_transpose(inputs=net, filters=32, kernel_size=(5, 5), strides=(2, 2), use_bias=use_bias, padding='same',
                                        kernel_initializer=xavier_init_conv, name='layer6/trconv', reuse=reuse)
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=-1, name='layer6/batchnorm', reuse=reuse)
        net = tf.nn.leaky_relu(net, alpha=0.15, name='layer6/act')
        
        net = tf.layers.conv2d_transpose(inputs=net, filters=16, kernel_size=(5, 5), strides=(2, 2), use_bias=use_bias, padding='same',
                                        kernel_initializer=xavier_init_conv, name='layer7/trconv', reuse=reuse)
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=-1, name='layer7/batchnorm', reuse=reuse)
        net = tf.nn.leaky_relu(net, alpha=0.15, name='layer7/act')
        
        net = tf.layers.conv2d_transpose(inputs=net, filters=8, kernel_size=(5, 5), strides=(1, 1), use_bias=use_bias, padding='same',
                                        kernel_initializer=xavier_init_conv, name='layer8/trconv', reuse=reuse)
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=-1, name='layer8/batchnorm', reuse=reuse)
        net = tf.nn.leaky_relu(net, alpha=0.15, name='layer8/act')
        
        net = tf.layers.conv2d_transpose(inputs=net, filters=x_in.shape[-1], kernel_size=(5, 5), strides=(1, 1), use_bias=use_bias, padding='same',
                                        kernel_initializer=xavier_init_conv, name='layer9/trconv', reuse=reuse)
    return tf.nn.sigmoid(net)
        
        
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]
        

M = 5
z_dim = 50

is_train = tf.placeholder(dtype=tf.bool, name='is_train')
x_in     = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
z_in     = tf.placeholder(dtype=tf.float32, shape=[None, z_dim])

G_out    = generator(z_in)
D_real   = autoencoder(x_in)
D_fake   = autoencoder(G_out)

d_loss_real = tf.reduce_sum(tf.square(x_in-D_real), axis=(1, 2, 3))
d_loss_fake = tf.nn.relu(M - tf.reduce_sum(tf.square(G_out-D_fake), axis=(1, 2, 3)))

d_loss = tf.reduce_mean(d_loss_real) + tf.reduce_mean(d_loss_fake)
g_loss = tf.reduce_mean(tf.reduce_sum(tf.square(G_out-D_fake), axis=(1, 2, 3)))

g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
g_update_ops = [var for var in update_ops if 'generator' in var.name]
d_update_ops = [var for var in update_ops if 'discriminator' in var.name]


with tf.control_dependencies(d_update_ops):
    d_opt1 = tf.train.AdamOptimizer(learning_rate=1E-3).minimize(loss=d_loss, var_list=d_vars)
    d_opt2 = tf.train.AdamOptimizer(learning_rate=1E-4).minimize(loss=d_loss, var_list=d_vars)
    
with tf.control_dependencies(g_update_ops):
    g_opt1 = tf.train.AdamOptimizer(learning_rate=1E-3).minimize(loss=g_loss, var_list=g_vars)
    g_opt2 = tf.train.AdamOptimizer(learning_rate=1E-4).minimize(loss=g_loss, var_list=g_vars)
    
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

train_dat = mnist.train.images
n_train = len(train_dat)

max_epoch = 500
minibatch_size = 256

pbar = tqdm(range(max_epoch))

d_opt, g_opt = d_opt1, g_opt1
g_loss_traj, d_loss_traj = [], []
for epoch in pbar:
    train_idx = np.arange(n_train)
    np.random.shuffle(train_idx)
    train_batch = chunks(train_idx, minibatch_size)
    
    if epoch == 300:
        d_opt, g_opt = d_opt2, g_opt2
        
    g_loss_stack, d_loss_stack = [], []
    for batch_idx in train_batch:
        batch_x = train_dat[batch_idx]
        batch_z = np.random.uniform(-1, 1, size=[len(batch_idx), z_dim])
        D_loss, _ = sess.run([d_loss, d_opt], feed_dict={x_in: batch_x, z_in: batch_z, is_train: True})
        G_loss, _ = sess.run([g_loss, g_opt], feed_dict={z_in: batch_z, is_train: True})
        
        g_loss_stack.append(G_loss)
        d_loss_stack.append(D_loss)
        
    g_loss_traj.append(np.mean(g_loss_stack))
    d_loss_traj.append(np.mean(d_loss_stack))
    pbar.set_description('G-loss: {:.4f} | D-loss: {:.4f} '.format(np.mean(g_loss_stack), np.mean(d_loss_stack)))
    
batch_z = np.random.uniform(-1, 1, size=[36, z_dim])
samples = sess.run(G_out, feed_dict={z_in: batch_z, is_train: False})

plt.figure(figsize=(10, 10))
for i, sample in enumerate(samples):
    plt.subplot(6, 6, i+1)
    plt.imshow(sample.reshape(28, 28), cmap='gray')
        