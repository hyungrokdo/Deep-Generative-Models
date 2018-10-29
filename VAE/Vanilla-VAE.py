# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:41:15 2018

@author: Hyungrok Do
         hyungrok.do11@gmail.com
         https://github.com/hyungrokdo
         
         A tensorflow impelementation of Gaussian encoder - Gaussian decoder variational autoencoder (Kingma and Welling, 2013)
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=False, reshape=False)
input_width, input_height, input_channel = mnist.train.images[0].shape
z_dim = 50

def encoder(x_in, use_batchnorm=True, use_bias=True):
    reuse = tf.AUTO_REUSE
    xavier_init, xavier_init_conv = tf.contrib.layers.xavier_initializer(uniform=True), tf.contrib.layers.xavier_initializer_conv2d(uniform=True)
    with tf.variable_scope('encoder', reuse=reuse):
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
        
        net = tf.layers.flatten(net, name='layer3/flatten')
        
        z_mean   = tf.layers.dense(inputs=net, units=z_dim, kernel_initializer=xavier_init, use_bias=use_bias, name='layer4/z_mean')
        z_logvar = tf.layers.dense(inputs=net, units=z_dim, kernel_initializer=xavier_init, use_bias=use_bias, name='layer4/z_logvar')
        
    return z_mean, z_logvar

def decoder(z_in, use_batchnorm=True, use_bias=True):
    reuse = tf.AUTO_REUSE
    xavier_init, xavier_init_conv = tf.contrib.layers.xavier_initializer(uniform=True), tf.contrib.layers.xavier_initializer_conv2d(uniform=True)
    with tf.variable_scope('decoder', reuse=reuse):
        net = tf.layers.dense(inputs=z_in, units=7*7*64, kernel_initializer=xavier_init, use_bias=use_bias, name='layer1/dense', reuse=reuse)
        net = tf.reshape(net, (-1, 7, 7, 64), name='layer1/reshape')
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=3, name='layer1/batchnorm', reuse=reuse)
        net = tf.nn.leaky_relu(net, name='layer1/act')
        
        net = tf.layers.conv2d_transpose(inputs=net, filters=32, kernel_size=(3, 3), strides=(2, 2), use_bias=use_bias, padding='same',
                                         kernel_initializer=xavier_init_conv, name='layer2/convtr', reuse=reuse)
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=3, name='layer2/batchnorm', reuse=reuse)
        net = tf.nn.leaky_relu(net, name='layer2/act')
        
        net = tf.layers.conv2d_transpose(inputs=net, filters=16, kernel_size=(3, 3), strides=(2, 2), use_bias=use_bias, padding='same',
                                         kernel_initializer=xavier_init_conv, name='layer3/convtr', reuse=reuse)
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=3, name='layer3/batchnorm', reuse=reuse)
        net = tf.nn.leaky_relu(net, name='layer3/act')
        
        x_mean   = tf.layers.conv2d_transpose(inputs=net, filters=1, kernel_size=(1, 1), strides=(1, 1), use_bias=use_bias, padding='same',
                                            kernel_initializer=xavier_init_conv, name='layer4/x_mean', reuse=reuse)
        x_logvar = tf.layers.conv2d_transpose(inputs=net, filters=1, kernel_size=(1, 1), strides=(1, 1), use_bias=use_bias, padding='same',
                                            kernel_initializer=xavier_init_conv, name='layer4/x_logvar', reuse=reuse)
        
    return x_mean, x_logvar

def sample_from_gaussian(mean, logvar):
    eps = tf.random_normal(shape=tf.shape(mean))
    return mean + tf.exp(logvar/2)*eps

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]
        
minibatch_size = 100
max_epoch = 100

is_train           = tf.placeholder(dtype=tf.bool, name='is_train')
x_in               = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
z_mean, z_logvar   = encoder(x_in)
z_sample           = sample_from_gaussian(z_mean, z_logvar)
x_mean, x_logvar   = decoder(z_sample)

z_in               = tf.stop_gradient(tf.placeholder(dtype=tf.float32, shape=[None, z_dim]))
x_mean_, x_logvar_ = decoder(z_in)

kl_divergence  = -0.5*tf.reduce_sum(1+z_logvar-tf.square(z_mean)-tf.exp(z_logvar), axis=1)
log_recon_prob = -0.5*(tf.reduce_sum(tf.squared_difference(x_in, x_mean)/tf.exp(x_logvar), axis=[1, 2, 3]) + tf.log(tf.reduce_sum(tf.exp(x_logvar), axis=[1, 2, 3])))

vae_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
vae_loss = tf.reduce_mean(kl_divergence - log_recon_prob)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    vae_opt  = tf.train.AdamOptimizer(learning_rate=1E-3).minimize(loss=vae_loss, var_list=vae_vars)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

train_dat = mnist.train.images
n_train = len(train_dat)

max_epoch = 200
minibatch_size = 256

pbar = tqdm(range(max_epoch))

for epoch in pbar:
    train_idx = np.arange(n_train)
    np.random.shuffle(train_idx)
    train_batch = chunks(train_idx, minibatch_size)
    
    loss_stack, kld_stack, log_recon_prob_stack = [], [], []
    for batch_idx in train_batch:
        batch_dat = train_dat[batch_idx]
        batch_loss, batch_kld, batch_log_recon_prob, _ = sess.run([vae_loss, kl_divergence, log_recon_prob, vae_opt],
                                                                  feed_dict={x_in: batch_dat, is_train: True})
        loss_stack.append(batch_loss)
        kld_stack += batch_kld.tolist()
        log_recon_prob_stack += batch_log_recon_prob.tolist()
        
    pbar.set_description('Loss: {:.4f} | KLD: {:.4f} | Log-Recon-Prob: {:.4f}'.format(np.mean(loss_stack), np.mean(kld_stack), np.mean(log_recon_prob_stack)))

batch_z = np.random.uniform(-20, 20, size=[16, z_dim])
samples = sess.run(x_mean_, feed_dict={z_in: batch_z, is_train: False})
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for i, sample in enumerate(samples):
    plt.subplot(4, 4, i+1)
    plt.imshow(sample.reshape(28, 28), cmap='gray')