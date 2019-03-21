# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:41:15 2018

@author: Hyungrok Do
         hyungrok.do11@gmail.com
         https://github.com/hyungrokdo
         
         A tensorflow impelementation of Gaussian encoder - Gaussian decoder adversarial autoencoder (Kingma and Welling, 2013)
"""

import matplotlib.pyplot as plt 
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=False, reshape=False)
z_dim = 25

def encoder(x_in, use_batchnorm=True, use_bias=True):
    reuse = tf.AUTO_REUSE
    xavier_init, xavier_init_conv = tf.contrib.layers.xavier_initializer(uniform=False), tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
    with tf.variable_scope('aae/encoder', reuse=reuse):
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
        
        mean   = tf.layers.dense(inputs=net, units=z_dim, kernel_initializer=xavier_init, use_bias=use_bias, name='layer4/z_mean')
        logvar = tf.layers.dense(inputs=net, units=z_dim, kernel_initializer=xavier_init, use_bias=use_bias, name='layer4/z_logvar')
        
    return mean, logvar

def decoder(z_in, use_batchnorm=True, use_bias=True):
    reuse = tf.AUTO_REUSE
    xavier_init, xavier_init_conv = tf.contrib.layers.xavier_initializer(uniform=False), tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
    with tf.variable_scope('aae/decoder', reuse=reuse):
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
        
    return x_mean, tf.clip_by_value(x_logvar, -3, 3)

def discriminator(z_in, use_bias=True):
    reuse = tf.AUTO_REUSE
    xavier_init = tf.contrib.layers.xavier_initializer(uniform=False)
    with tf.variable_scope('aae/discriminator', reuse=reuse):
        net = tf.layers.dense(inputs=z_in, units=25, kernel_initializer=xavier_init, use_bias=use_bias, name='layer1/dense', reuse=reuse)
        net = tf.nn.leaky_relu(net, name='layer1/act')
        
        net = tf.layers.dense(inputs=net, units=20, kernel_initializer=xavier_init, use_bias=use_bias, name='layer2/dense', reuse=reuse)
        net = tf.nn.leaky_relu(net, name='layer2/act')
        
        net = tf.layers.dense(inputs=net, units=15, kernel_initializer=xavier_init, use_bias=use_bias, name='layer3/dense', reuse=reuse)
        net = tf.nn.leaky_relu(net, name='layer3/act')
        
        net = tf.layers.dense(inputs=net, units=1, kernel_initializer=xavier_init, use_bias=use_bias, name='layer4/dense', reuse=reuse)
    return net
    
def sample_from_gaussian(mean, logvar):
    eps = tf.random_normal(shape=tf.shape(mean))
    return mean + tf.exp(logvar/2)*eps

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

is_train           = tf.placeholder(dtype=tf.bool, name='is_train')
x_in               = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
z_mean, z_logvar   = encoder(x_in)
z_enc_out          = sample_from_gaussian(z_mean, z_logvar)
z_dis_in           = tf.placeholder(dtype=tf.float32, shape=[None, z_dim])
d_enc_out          = discriminator(z_enc_out)
d_target           = discriminator(z_dis_in)
x_mean, x_logvar   = decoder(z_enc_out)

d_loss_enc         = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_enc_out, labels=tf.zeros_like(d_enc_out))
d_loss_target      = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_target, labels=tf.ones_like(d_target))
g_loss             = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_enc_out, labels=tf.ones_like(d_enc_out)))
d_loss             = tf.reduce_mean(d_loss_enc) + tf.reduce_mean(d_loss_target)

log_recon_likelihood = -0.5*(tf.reduce_sum(tf.squared_difference(x_in, x_mean)*tf.exp(-x_logvar), axis=[1, 2, 3]) + tf.reduce_sum(x_logvar, axis=[1, 2, 3]) + 784*np.log(2*np.pi))
ae_loss = -tf.reduce_mean(log_recon_likelihood)

enc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='aae/encoder')
dec_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='aae/decoder')
d_vars   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='aae/discriminator')

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    ae_opt  = tf.train.AdamOptimizer(learning_rate=1E-4).minimize(loss=ae_loss, var_list=enc_vars+dec_vars)
    g_opt   = tf.train.AdamOptimizer(learning_rate=1E-4).minimize(loss=g_loss, var_list=enc_vars)

d_opt   = tf.train.AdamOptimizer(learning_rate=1E-4).minimize(loss=d_loss, var_list=d_vars)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

train_dat = mnist.train.images
n_train = len(train_dat)

max_epoch = 100
minibatch_size = 256

pbar = tqdm(range(max_epoch))
ae_loss_traj, d_loss_traj, g_loss_traj = [], [], []
for epoch in pbar:
    train_idx = np.arange(n_train)
    np.random.shuffle(train_idx)
    train_batch = chunks(train_idx, minibatch_size)

    ae_loss_stack, d_loss_stack, g_loss_stack = [], [], []
    for batch_idx in train_batch:
        batch_dat    = train_dat[batch_idx]
        batch_target = np.random.multivariate_normal(np.zeros(z_dim), np.identity(z_dim), size=len(batch_dat))
        
        sess.run(ae_opt, feed_dict={x_in: batch_dat, is_train: True})
        sess.run(d_opt, feed_dict={x_in: batch_dat, z_dis_in: batch_target, is_train: True})
        sess.run(g_opt, feed_dict={x_in: batch_dat, is_train: True})
        
        batch_ae_loss, batch_d_loss, batch_g_loss = sess.run([ae_loss, d_loss, g_loss],
                                                              feed_dict={x_in: batch_dat, z_dis_in: batch_target, is_train: True})
        ae_loss_stack.append(batch_ae_loss)
        d_loss_stack.append(batch_d_loss)
        g_loss_stack.append(batch_g_loss)

    ae_loss_traj.append(np.mean(ae_loss_stack))        
    d_loss_traj.append(np.mean(d_loss_stack))
    g_loss_traj.append(np.mean(g_loss_stack))
    
    pbar.set_description('AE Loss: {:.4f} | D Loss: {:.4f} | G Loss: {:.4f}'.format(np.mean(ae_loss_stack),
                                                                                    np.mean(d_loss_stack), np.mean(g_loss_traj)))

batch_z, z_logvar_ = sess.run([z_mean, z_logvar], feed_dict={x_in:train_dat[np.random.choice(n_train, 16)], is_train: False})
samples = sess.run(x_mean, feed_dict={z_enc_out: batch_z, is_train: False})
plt.figure(figsize=(10, 10))
for i, sample in enumerate(samples):
    plt.subplot(4, 4, i+1)
    plt.imshow(sample.reshape(28, 28), cmap='gray')
plt.show()

        
