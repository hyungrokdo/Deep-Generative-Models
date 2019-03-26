# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 20:56:49 2019

@author: Hyungrok Do
         hyungrok.do11@gmail.com
         https://github.com/hyungrokdo
         
         A tensorflow-layer API implementation of Wasserstein GAN (WGAN)
         
         Arjovsky, M., Chintala, S., & Bottou, L. (2017).
         Wasserstein gan. arXiv preprint arXiv:1701.07875.
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
        
def discriminator(x_in, use_bias=True):
    reuse = tf.AUTO_REUSE
    xavier_init_conv = tf.contrib.layers.xavier_initializer_conv2d(uniform=True)
    with tf.variable_scope('discriminator', reuse=reuse):
        net = tf.layers.conv2d(inputs=x_in, filters=8, kernel_size=(5, 5), strides=(2, 2), use_bias=use_bias, padding='same',
                               kernel_initializer=xavier_init_conv, name='layer1/conv', reuse=reuse)
        net = tf.nn.leaky_relu(net, name='layer1/act')
        
        net = tf.layers.conv2d(inputs=net, filters=16, kernel_size=(5, 5), strides=(2, 2), use_bias=use_bias, padding='same',
                               kernel_initializer=xavier_init_conv, name='layer2/conv', reuse=reuse)
        net = tf.nn.leaky_relu(net, name='layer2/act')
        
        net = tf.layers.flatten(net, name='layer2/flatten')
        
        net = tf.layers.dense(inputs=net, units=50, use_bias=use_bias, name='layer3/dense')
        net = tf.nn.leaky_relu(net, name='layer3/act')
        
        net = tf.layers.dense(inputs=net, units=1, use_bias=use_bias, name='layer4/output')
    return net

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

z_dim = 50
is_train = tf.placeholder(tf.bool, name='is_train')
z = tf.placeholder(dtype=tf.float32, shape=[None, z_dim], name='z')
x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
G = generator(z)
D_real, D_fake = discriminator(x), discriminator(G)

d_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)
g_loss = -tf.reduce_mean(D_fake)

g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
g_update_ops = [var for var in update_ops if 'generator' in var.name]

d_opt1       = tf.train.AdamOptimizer(learning_rate=1E-4)
d_grad1      = d_opt1.compute_gradients(loss=d_loss, var_list=d_vars)
d_grad_clip1 = [(tf.clip_by_value(grad, -.01, .01), var) for grad, var in d_grad1]
d_opt1       = d_opt1.apply_gradients(d_grad_clip1)

d_opt2       = tf.train.AdamOptimizer(learning_rate=1E-5)
d_grad2      = d_opt2.compute_gradients(loss=d_loss, var_list=d_vars)
d_grad_clip2 = [(tf.clip_by_value(grad, -.01, .01), var) for grad, var in d_grad2]
d_opt2       = d_opt2.apply_gradients(d_grad_clip2)
    
with tf.control_dependencies(g_update_ops):
    g_opt1       = tf.train.AdamOptimizer(learning_rate=1E-4)
    g_grad1      = g_opt1.compute_gradients(loss=g_loss, var_list=g_vars)
    g_grad_clip1 = [(tf.clip_by_value(grad, -.01, .01), var) for grad, var in g_grad1 if 'batchnorm' not in var.name]
    g_opt1       = g_opt1.apply_gradients(g_grad_clip1)
    
    g_opt2       = tf.train.AdamOptimizer(learning_rate=1E-5)
    g_grad2      = g_opt2.compute_gradients(loss=g_loss, var_list=g_vars)
    g_grad_clip2 = [(tf.clip_by_value(grad, -.01, .01), var) for grad, var in g_grad2 if 'batchnorm' not in var.name]
    g_opt2       = g_opt2.apply_gradients(g_grad_clip2)
    
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

train_dat = mnist.train.images
n_train = len(train_dat)

max_epoch = 400
minibatch_size = 256

pbar = tqdm(range(max_epoch))

d_opt, g_opt = d_opt1, g_opt1
g_loss_traj, d_loss_traj = [], []
for epoch in pbar:
    train_idx = np.arange(n_train)
    np.random.shuffle(train_idx)
    train_batch = chunks(train_idx, minibatch_size)
    
    if epoch == 200:
        d_opt, g_opt = d_opt2, g_opt2
        
    g_loss_stack, d_loss_stack = [], []
    for batch_idx in train_batch:
        batch_x = train_dat[batch_idx]
        batch_z = np.random.uniform(-1, 1, size=[len(batch_idx), z_dim])
        
        sess.run(d_opt, feed_dict={x: batch_x, z: batch_z, is_train: True})
        sess.run(g_opt, feed_dict={z: batch_z, is_train: True})
       
        D_loss, G_loss = sess.run([d_loss, g_loss], feed_dict={x: batch_x, z: batch_z, is_train: True})

        g_loss_stack.append(G_loss)
        d_loss_stack.append(D_loss)
        
    g_loss_traj.append(np.mean(g_loss_stack))
    d_loss_traj.append(np.mean(d_loss_stack))
    pbar.set_description('G-loss: {:.4f} | D-loss: {:.4f}'.format(np.mean(g_loss_stack), np.mean(d_loss_stack)))
    
batch_z = np.random.uniform(-1, 1, size=[36, z_dim])
samples = sess.run(G, feed_dict={z: batch_z, is_train: False})
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for i, sample in enumerate(samples):
    plt.subplot(6, 6, i+1)
    plt.imshow(sample.reshape(28, 28), cmap='gray')
        
