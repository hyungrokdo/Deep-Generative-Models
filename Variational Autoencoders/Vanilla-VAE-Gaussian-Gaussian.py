# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:41:15 2018

@author: Hyungrok Do
         hyungrok.do11@gmail.com
         https://github.com/hyungrokdo
         
         A tensorflow impelementation of Gaussian encoder - Gaussian decoder variational autoencoder (Kingma and Welling, 2013)
"""

import matplotlib.pyplot as plt 
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/MNIST', one_hot=False, reshape=False)
z_dim = 25

def encoder(x_in, use_batchnorm=True, use_bias=True):
    reuse = tf.AUTO_REUSE
    xavier_init, xavier_init_conv = tf.contrib.layers.xavier_initializer(uniform=False), tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
    with tf.variable_scope('vae/encoder', reuse=reuse):
        net = tf.layers.conv2d(inputs=x_in, filters=8, kernel_size=(5, 5), strides=(1, 1), use_bias=use_bias, padding='same',
                               kernel_initializer=xavier_init_conv, name='layer1/conv', reuse=reuse)
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=3, name='layer1/batchnorm', reuse=reuse)
        net = tf.nn.leaky_relu(net, name='layer1/act')
        
        net = tf.layers.conv2d(inputs=net, filters=16, kernel_size=(5, 5), strides=(2, 2), use_bias=use_bias, padding='same',
                               kernel_initializer=xavier_init_conv, name='layer2/conv', reuse=reuse)
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=3, name='layer2/batchnorm', reuse=reuse)
        net = tf.nn.leaky_relu(net, name='layer2/act')
        
        net = tf.layers.conv2d(inputs=net, filters=32, kernel_size=(5, 5), strides=(2, 2), use_bias=use_bias, padding='same',
                               kernel_initializer=xavier_init_conv, name='layer3/conv', reuse=reuse)
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=3, name='layer3/batchnorm', reuse=reuse)
        net = tf.nn.leaky_relu(net, name='layer3/act')
        
        net = tf.layers.flatten(net, name='layer4/flatten')
        
        z_mean   = tf.layers.dense(inputs=net, units=z_dim, kernel_initializer=xavier_init, use_bias=use_bias, name='layer4/z_mean')
        z_logvar = tf.layers.dense(inputs=net, units=z_dim, kernel_initializer=xavier_init, use_bias=use_bias, name='layer4/z_logvar')
        
    return z_mean, z_logvar

def decoder(z_in, use_batchnorm=True, use_bias=True):
    reuse = tf.AUTO_REUSE
    xavier_init, xavier_init_conv = tf.contrib.layers.xavier_initializer(uniform=False), tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
    with tf.variable_scope('vae/decoder', reuse=reuse):
        net = tf.layers.dense(inputs=z_in, units=7*7*32, kernel_initializer=xavier_init, use_bias=use_bias, name='layer1/dense', reuse=reuse)
        net = tf.reshape(net, (-1, 7, 7, 32), name='layer1/reshape')
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=3, name='layer1/batchnorm', reuse=reuse)
        net = tf.nn.relu(net, name='layer1/act')
        
        net = tf.layers.conv2d_transpose(inputs=net, filters=16, kernel_size=(5, 5), strides=(2, 2), use_bias=use_bias, padding='same',
                                         kernel_initializer=xavier_init_conv, name='layer2/convtr', reuse=reuse)
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=3, name='layer2/batchnorm', reuse=reuse)
        net = tf.nn.relu(net, name='layer2/act')
        
        net = tf.layers.conv2d_transpose(inputs=net, filters=8, kernel_size=(5, 5), strides=(2, 2), use_bias=use_bias, padding='same',
                                         kernel_initializer=xavier_init_conv, name='layer3/convtr', reuse=reuse)
        if use_batchnorm:
            net = tf.layers.batch_normalization(inputs=net, training=is_train, axis=3, name='layer3/batchnorm', reuse=reuse)
        net = tf.nn.relu(net, name='layer3/act')

        x_mean   = tf.layers.conv2d_transpose(inputs=net, filters=1, kernel_size=(5, 5), strides=(1, 1), use_bias=use_bias, padding='same',
                                              kernel_initializer=xavier_init_conv, name='layer4/x_mean', reuse=reuse)
        x_logvar = tf.layers.conv2d_transpose(inputs=net, filters=1, kernel_size=(5, 5), strides=(1, 1), use_bias=use_bias, padding='same',
                                              kernel_initializer=xavier_init_conv, name='layer4/x_logvar', reuse=reuse)
        x_logvar = tf.clip_by_value(x_logvar, -2, 10, name='layer4/x_logvar_clipped')
    return x_mean, x_logvar

def sample_from_gaussian(mean, logvar):
    eps = tf.random_normal(shape=tf.shape(mean))
    return mean + tf.exp(logvar/2)*eps

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

is_train           = tf.placeholder(dtype=tf.bool, name='is_train')
x_in               = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
z_mean, z_logvar   = encoder(x_in)
z_sample           = sample_from_gaussian(z_mean, z_logvar)
x_mean, x_logvar   = decoder(z_sample)

kl_divergence  = -0.5*tf.reduce_sum(1+z_logvar-tf.square(z_mean)-tf.exp(z_logvar), axis=1)
log_recon_likelihood = -0.5*(tf.reduce_sum(tf.squared_difference(x_in, x_mean)*tf.exp(-x_logvar), axis=[1, 2, 3]) + tf.reduce_sum(x_logvar, axis=[1, 2, 3]) + 784*np.log(2*np.pi))

vae_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vae')
vae_loss = tf.reduce_mean(kl_divergence - log_recon_likelihood)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    vae_opt  = tf.train.AdamOptimizer(learning_rate=1E-4).minimize(loss=vae_loss, var_list=vae_vars)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

train_dat = mnist.train.images
n_train = len(train_dat)

max_epoch = 100
minibatch_size = 512

pbar = tqdm(range(max_epoch))
loss_traj, kld_traj, log_recon_likelihood_traj = [], [], []
for epoch in pbar:
    train_idx = np.arange(n_train)
    np.random.shuffle(train_idx)
    train_batch = chunks(train_idx, minibatch_size)

    loss_stack, kld_stack, log_recon_likelihood_stack = [], [], []
    for batch_idx in train_batch:
        batch_dat = train_dat[batch_idx]
        batch_loss, batch_kld, batch_log_recon_likelihood, _ = sess.run([vae_loss, kl_divergence, log_recon_likelihood, vae_opt],
                                                                  feed_dict={x_in: batch_dat, is_train: True})
        loss_stack.append(batch_loss)
        kld_stack += batch_kld.tolist()
        log_recon_likelihood_stack += batch_log_recon_likelihood.tolist()

    loss_traj.append(np.mean(loss_stack))        
    kld_traj.append(np.mean(kld_stack))
    log_recon_likelihood_traj.append(np.mean(log_recon_likelihood_stack))
    
    pbar.set_description('Loss: {:.4f} | KLD: {:.4f} | Log-Recon-Likelihood: {:.4f}'.format(np.mean(loss_stack),
                                                                                      np.mean(kld_stack), np.mean(log_recon_likelihood_stack)))

batch_z, z_logvar_ = sess.run([z_mean, z_logvar], feed_dict={x_in:train_dat[np.random.choice(n_train, 16)], is_train: False})
samples = sess.run(x_mean, feed_dict={z_sample: batch_z, is_train: False})

plt.figure(figsize=(10, 10))
for i, sample in enumerate(samples):
    plt.subplot(4, 4, i+1)
    plt.imshow(sample.reshape(28, 28), cmap='gray')
plt.show()

        
