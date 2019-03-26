# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:54:51 2019

@author: Hyungrok Do
         hyungrok.do11@gmail.com
         https://github.com/hyungrokdo
         
         A tensorflow-layer API implementation of Bidirectional GAN (BiGAN)
         
         Donahue, J., Krähenbühl, P., & Darrell, T. (2016).
         Adversarial feature learning. arXiv preprint arXiv:1605.09782.
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=False, reshape=False)

z_dim = 50

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

def encoder(x_in, use_batchnorm=True, use_bias=True):
    reuse = tf.AUTO_REUSE
    xavier_init_conv = tf.contrib.layers.xavier_initializer_conv2d(uniform=True)
    with tf.variable_scope('encoder', reuse=reuse):
        net = tf.layers.conv2d(inputs=x_in, filters=8, kernel_size=(5, 5), strides=(2, 2), use_bias=use_bias, padding='same',
                               kernel_initializer=xavier_init_conv, name='layer1/conv', reuse=reuse)
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=-1, name='layer1/batchnorm', reuse=reuse)
        net = tf.nn.leaky_relu(net, name='layer1/act')
        
        net = tf.layers.conv2d(inputs=net, filters=16, kernel_size=(5, 5), strides=(2, 2), use_bias=use_bias, padding='same',
                               kernel_initializer=xavier_init_conv, name='layer2/conv', reuse=reuse)
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=-1, name='layer2/batchnorm', reuse=reuse)
        net = tf.nn.leaky_relu(net, name='layer2/act')
        
        net = tf.layers.flatten(net, name='layer2/flatten')
        
        net = tf.layers.dense(inputs=net, units=50, name='layer3/dense')
        net = tf.nn.leaky_relu(net, name='layer3/act')
        
        net = tf.layers.dense(inputs=net, units=z_dim, name='layer4/output')
        
    return net
        
def discriminator(x_in, z_in, use_bias=True):
    reuse = tf.AUTO_REUSE
    xavier_init_conv = tf.contrib.layers.xavier_initializer_conv2d(uniform=True)
    with tf.variable_scope('discriminator/x_part', reuse=reuse):
        net_x = tf.layers.conv2d(inputs=x_in, filters=8, kernel_size=(5, 5), strides=(2, 2), use_bias=use_bias, padding='same',
                                 kernel_initializer=xavier_init_conv, name='layer1/conv', reuse=reuse)
        net_x = tf.nn.leaky_relu(net_x, name='layer1/act')
        
        net_x = tf.layers.conv2d(inputs=net_x, filters=16, kernel_size=(5, 5), strides=(2, 2), use_bias=use_bias, padding='same',
                                 kernel_initializer=xavier_init_conv, name='layer2/conv', reuse=reuse)
        net_x = tf.nn.leaky_relu(net_x, name='layer2/act')
        
        net_x = tf.layers.flatten(net_x, name='layer2/flatten')
        
        net_x = tf.layers.dense(inputs=net_x, units=50, name='layer3/dense')
        
    with tf.variable_scope('discriminator/z_part', reuse=reuse):
        net_z = tf.layers.dense(inputs=z_in, units=50, name='layer1/dense')
        
    with tf.variable_scope('discriminator/common_part', reuse=reuse):
        net = tf.concat([net_x, net_z], axis=1, name='layer1/xz_concat')
        
        net = tf.layers.dense(inputs=net, units=50, name='layer2/dense')
        net = tf.nn.leaky_relu(net, alpha=0.15, name='layer2/act')
        
        net = tf.layers.dense(inputs=net, units=1, name='layer3/output')
        
    return net

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

is_train = tf.placeholder(tf.bool, name='is_train')
z_fake = tf.placeholder(dtype=tf.float32, shape=[None, z_dim], name='z')
x      = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
G      = generator(z_fake)
z_real = encoder(x)
D_real = discriminator(x, z_real)
D_fake = discriminator(G, z_fake)

e_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.zeros_like(D_real))
d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real))
d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake))

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))
d_loss = tf.reduce_mean(d_loss_real) + tf.reduce_mean(d_loss_fake)
e_loss = tf.reduce_mean(e_loss_real)

d_acc = tf.reduce_mean(tf.cast(tf.equal(tf.concat([tf.ones_like(D_real, tf.int32), tf.zeros_like(D_fake, tf.int32)], 0),
                                tf.concat([tf.cast(tf.greater(D_real, 0.5), tf.int32), tf.cast(tf.greater(D_fake, 0.5), tf.int32)], 0)), tf.float32))

g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
e_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
g_update_ops = [var for var in update_ops if 'generator' in var.name]
e_update_ops = [var for var in update_ops if 'encoder' in var.name]

d_opt1 = tf.train.AdamOptimizer(learning_rate=1E-3).minimize(loss=d_loss, var_list=d_vars)
d_opt2 = tf.train.AdamOptimizer(learning_rate=1E-4).minimize(loss=d_loss, var_list=d_vars)

with tf.control_dependencies(e_update_ops):
    e_opt1 = tf.train.AdamOptimizer(learning_rate=1E-3).minimize(loss=e_loss, var_list=e_vars)
    e_opt2 = tf.train.AdamOptimizer(learning_rate=1E-4).minimize(loss=e_loss, var_list=e_vars)
    
with tf.control_dependencies(g_update_ops):
    g_opt1 = tf.train.AdamOptimizer(learning_rate=1E-3).minimize(loss=g_loss, var_list=g_vars)
    g_opt2 = tf.train.AdamOptimizer(learning_rate=1E-4).minimize(loss=g_loss, var_list=g_vars)
    
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

train_dat = mnist.train.images
n_train = len(train_dat)

max_epoch = 400
minibatch_size = 256

pbar = tqdm(range(max_epoch))

d_opt, g_opt, e_opt = d_opt1, g_opt1, e_opt1
g_loss_traj, d_loss_traj = [], []
for epoch in pbar:
    train_idx = np.arange(n_train)
    np.random.shuffle(train_idx)
    train_batch = chunks(train_idx, minibatch_size)
    
    if epoch == 200:
        d_opt, g_opt, e_opt = d_opt2, g_opt2, e_opt2
        
    g_loss_stack, d_loss_stack, d_acc_stack = [], [], []
    for batch_idx in train_batch:
        batch_x = train_dat[batch_idx]
        batch_z = np.random.uniform(-1, 1, size=[len(batch_idx), z_dim])
        
        sess.run(e_opt, feed_dict={x: batch_x, is_train: True})
        sess.run(g_opt, feed_dict={z_fake: batch_z, is_train: True})
        sess.run(d_opt, feed_dict={x: batch_x, z_fake: batch_z, is_train: True})
        
        G_loss, D_loss, D_acc = sess.run([g_loss, d_loss, d_acc], feed_dict={x: batch_x, z_fake: batch_z, is_train: True})
        
        g_loss_stack.append(G_loss)
        d_loss_stack.append(D_loss)
        d_acc_stack.append(D_acc)
        
    g_loss_traj.append(np.mean(g_loss_stack))
    d_loss_traj.append(np.mean(d_loss_stack))
    pbar.set_description('G-loss: {:.4f} | D-loss: {:.4f} | D-accuracy: {:.4f}'.format(np.mean(g_loss_stack), np.mean(d_loss_stack), np.mean(d_acc_stack)))
    
batch_z = np.random.uniform(-1, 1, size=[36, z_dim])
samples = sess.run(G, feed_dict={z_fake: batch_z, is_train: False})
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for i, sample in enumerate(samples):
    plt.subplot(6, 6, i+1)
    plt.imshow(sample.reshape(28, 28), cmap='gray')
        
