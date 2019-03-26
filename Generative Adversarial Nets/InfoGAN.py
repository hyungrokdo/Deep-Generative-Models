# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 14:06:32 2018

@author: Hyungrok Do
         hyungrok.do11@gmail.com
         https://github.com/hyungrokdo
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm

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
        
        net = tf.layers.conv2d_transpose(inputs=net, filters=32, kernel_size=(5, 5), strides=(2, 2), use_bias=use_bias, padding='same',
                                         kernel_initializer=xavier_init_conv, name='layer2/convtr', reuse=reuse)
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=3, name='layer2/batchnorm', reuse=reuse)
        net = tf.nn.leaky_relu(net, name='layer2/act')
        
        net = tf.layers.conv2d_transpose(inputs=net, filters=16, kernel_size=(5, 5), strides=(2, 2), use_bias=use_bias, padding='same',
                                         kernel_initializer=xavier_init_conv, name='layer3/convtr', reuse=reuse)
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=3, name='layer3/batchnorm', reuse=reuse)
        net = tf.nn.leaky_relu(net, name='layer3/act')
        
        net = tf.layers.conv2d_transpose(inputs=net, filters=1, kernel_size=(5, 5), strides=(1, 1), use_bias=use_bias, padding='same',
                                         kernel_initializer=xavier_init_conv, name='layer4/output', reuse=reuse)
    return tf.nn.sigmoid(net)
        
def discriminator(x_in, use_bias=True):
    reuse = tf.AUTO_REUSE
    xavier_init_conv = tf.contrib.layers.xavier_initializer_conv2d(uniform=True)
    with tf.variable_scope('discriminator', reuse=reuse):
        net = tf.layers.conv2d(inputs=x_in, filters=4, kernel_size=(5, 5), strides=(2, 2), use_bias=use_bias, padding='same',
                               kernel_initializer=xavier_init_conv, name='layer1/conv', reuse=reuse)
        net = tf.nn.leaky_relu(net, name='layer1/act')
        
        net = tf.layers.conv2d(inputs=net, filters=8, kernel_size=(5, 5), strides=(2, 2), use_bias=use_bias, padding='same',
                               kernel_initializer=xavier_init_conv, name='layer2/conv', reuse=reuse)
        net = tf.nn.leaky_relu(net, name='layer2/act')
        
        net = tf.layers.flatten(net, name='layer2/flatten')
        
        net = tf.layers.dense(inputs=net, units=50, name='layer3/dense')
        net = tf.nn.leaky_relu(net, name='layer3/act')
        
        d_logit = tf.layers.dense(inputs=net, units=1, name='layer4/d_logit')
        c_pred  = tf.layers.dense(inputs=net, units=c_dim, name='layer4/c_pred')
        
    return d_logit, c_pred

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

mi_loss_weight = 5
z_dim, c_dim = 35, 15
is_train = tf.placeholder(tf.bool, name='is_train')
z = tf.placeholder(dtype=tf.float32, shape=[None, z_dim], name='z')
c = tf.placeholder(dtype=tf.float32, shape=[None, c_dim], name='c')
x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='x')

G_in           = tf.concat([z, c], axis=1, name='G_in')
G_out          = generator(G_in)
D_real, _      = discriminator(x)
D_fake, c_pred = discriminator(G_out)

d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real))
d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake))
mi_loss     = tf.reduce_mean(tf.reduce_sum(tf.square(c-c_pred), axis=1)) * mi_loss_weight

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))
d_loss = tf.reduce_mean(d_loss_real) + tf.reduce_mean(d_loss_fake)

d_acc = tf.reduce_mean(tf.cast(tf.equal(tf.concat([tf.ones_like(D_real, tf.int32), tf.zeros_like(D_fake, tf.int32)], 0),
                                tf.concat([tf.cast(tf.greater(D_real, 0.5), tf.int32), tf.cast(tf.greater(D_fake, 0.5), tf.int32)], 0)), tf.float32))

g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
g_update_ops = [var for var in update_ops if 'generator' in var.name]

d_opt1 = tf.train.AdamOptimizer(learning_rate=1E-3).minimize(loss=d_loss, var_list=d_vars)
d_opt2 = tf.train.AdamOptimizer(learning_rate=1E-4).minimize(loss=d_loss, var_list=d_vars)
    
with tf.control_dependencies(g_update_ops):
    g_opt1 = tf.train.AdamOptimizer(learning_rate=1E-3).minimize(loss=g_loss, var_list=g_vars)
    g_opt2 = tf.train.AdamOptimizer(learning_rate=1E-4).minimize(loss=g_loss, var_list=g_vars)
    
    mi_opt1 = tf.train.AdamOptimizer(learning_rate=1E-3).minimize(loss=mi_loss, var_list=g_vars+d_vars)
    mi_opt2 = tf.train.AdamOptimizer(learning_rate=1E-4).minimize(loss=mi_loss, var_list=g_vars+d_vars)
    
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

train_dat = mnist.train.images
n_train = len(train_dat)

max_epoch = 500
minibatch_size = 256

pbar = tqdm(range(max_epoch))

d_opt, g_opt, mi_opt = d_opt1, g_opt1, mi_opt1
g_loss_traj, d_loss_traj = [], []
for epoch in pbar:
    train_idx = np.arange(n_train)
    np.random.shuffle(train_idx)
    train_batch = chunks(train_idx, minibatch_size)
    
    if epoch == 250:
        d_opt, g_opt, mi_opt = d_opt2, g_opt2, mi_opt2
        
    g_loss_stack, mi_loss_stack, d_loss_stack, d_acc_stack = [], [], [], []
    for batch_idx in train_batch:
        batch_x = train_dat[batch_idx]
        batch_z = np.random.uniform(-1, 1, size=[len(batch_idx), z_dim])
        batch_c = np.random.uniform(-1, 1, size=[len(batch_idx), c_dim])

        sess.run(d_opt, feed_dict={x: batch_x, z: batch_z, c: batch_c, is_train: False})
        sess.run(g_opt, feed_dict={z: batch_z, c:batch_c, is_train: True})
        sess.run(mi_opt, feed_dict={z: batch_z, c:batch_c, is_train: True})
        
        G_loss, MI_loss, D_loss, D_acc = sess.run([g_loss, mi_loss, d_loss, d_acc],
                                                  feed_dict={x: batch_x, z: batch_z, c: batch_c, is_train: False})
        
        g_loss_stack.append(G_loss)
        d_loss_stack.append(D_loss)
        mi_loss_stack.append(MI_loss)
        d_acc_stack.append(D_acc)
        
    g_loss_traj.append(np.mean(g_loss_stack))
    d_loss_traj.append(np.mean(d_loss_stack))
    pbar.set_description('G-loss: {:.4f} | MI-Loss: {:.4f} | D-loss: {:.4f} | D-accuracy: {:.4f}'.format(np.mean(g_loss_stack),
                         np.mean(mi_loss_stack), np.mean(d_loss_stack), np.mean(d_acc_stack)))
    
batch_z = np.repeat(np.random.uniform(-0.5, 0.5, size=(1, z_dim)), 36, axis=0).reshape(36, -1)
batch_c = np.repeat(np.random.uniform(-0.5, 0.5, size=(1, c_dim)), 36, axis=0).reshape(36, -1)
batch_c[:,6] = np.linspace(-1, 1, 36)
samples = sess.run(G_out, feed_dict={z: batch_z, c: batch_c, is_train: False})
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for i, sample in enumerate(samples):
    plt.subplot(6, 6, i+1)
    plt.imshow(sample.reshape(28, 28), cmap='gray')
        
