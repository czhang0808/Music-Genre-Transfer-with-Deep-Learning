import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
from utils import *


def batch_norm(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)


def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv
        return scale * normalized + offset


def conv2d(input_, output_dim, ks=7, s=2, stddev=0.02, padding='SAME', name="conv2d", sn = False):
    with tf.variable_scope(name):
        if sn:
            w = tf.get_variable(name = 'kernel', shape = [ks, ks, input_.get_shape()[-1], output_dim],
                                initializer = tf.truncated_normal_initializer(stddev=stddev), regularizer = None)
            return tf.nn.conv2d(input = input_, filter = spectral_normalization(w),
                             strides = [1, s, s, 1], padding = padding)
        else:
            return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                                weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                biases_initializer=None)


def deconv2d(input_, output_dim, ks=7, s=2, stddev=0.02, padding='SAME', name="deconv2d", sn = False):
    with tf.variable_scope(name):
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                    biases_initializer=None)


def relu(tensor_in):
    if tensor_in is not None:
        return tf.nn.relu(tensor_in)
    else:
        return tensor_in


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def to_binary_tf(bar_or_track_bar, threshold=0.0, track_mode=False, melody=False):
    """Return the binarize tensor of the input tensor (be careful of the channel order!)"""
    if track_mode:
        # melody track
        if melody:
            melody_is_max = tf.equal(bar_or_track_bar, tf.reduce_max(bar_or_track_bar, axis=2, keep_dims=True))
            melody_pass_threshold = (bar_or_track_bar > threshold)
            out_tensor = tf.logical_and(melody_is_max, melody_pass_threshold)
        # non-melody track
        else:
            out_tensor = (bar_or_track_bar > threshold)
        return out_tensor
    else:
        if len(bar_or_track_bar.get_shape()) == 4:
            melody_track = tf.slice(bar_or_track_bar, [0, 0, 0, 0], [-1, -1, -1, 1])
            other_tracks = tf.slice(bar_or_track_bar, [0, 0, 0, 1], [-1, -1, -1, -1])
        elif len(bar_or_track_bar.get_shape()) == 5:
            melody_track = tf.slice(bar_or_track_bar, [0, 0, 0, 0, 0], [-1, -1, -1, -1, 1])
            other_tracks = tf.slice(bar_or_track_bar, [0, 0, 0, 0, 1], [-1, -1, -1, -1, -1])
        # melody track
        melody_is_max = tf.equal(melody_track, tf.reduce_max(melody_track, axis=2, keep_dims=True))
        melody_pass_threshold = (melody_track > threshold)
        out_tensor_melody = tf.logical_and(melody_is_max, melody_pass_threshold)
        # other tracks
        out_tensor_others = (other_tracks > threshold)
        if len(bar_or_track_bar.get_shape()) == 4:
            return tf.concat([out_tensor_melody, out_tensor_others], 3)
        elif len(bar_or_track_bar.get_shape()) == 5:
            return tf.concat([out_tensor_melody, out_tensor_others], 4)


def to_chroma_tf(bar_or_track_bar, is_normalize=True):
    """Return the chroma tensor of the input tensor"""
    out_shape = tf.stack([tf.shape(bar_or_track_bar)[0], bar_or_track_bar.get_shape()[1], 12, 7,
                         bar_or_track_bar.get_shape()[3]])
    chroma = tf.reduce_sum(tf.reshape(tf.cast(bar_or_track_bar, tf.float32), out_shape), axis=3)
    if is_normalize:
        chroma_max = tf.reduce_max(chroma, axis=(1, 2, 3), keep_dims=True)
        chroma_min = tf.reduce_min(chroma, axis=(1, 2, 3), keep_dims=True)
        return tf.truediv(chroma - chroma_min, (chroma_max - chroma_min + 1e-15))
    else:
        return chroma


def to_binary(bars, threshold=0.0):
    """Turn velocity value into boolean"""
    track_is_max = tf.equal(bars, tf.reduce_max(bars, axis=-1, keep_dims=True))
    track_pass_threshold = (bars > threshold)
    out_track = tf.logical_and(track_is_max, track_pass_threshold)
    return out_track


def spectral_normalization(w, iteration = 1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u_l = tf.get_variable("u_l", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_bar = u_l
    v_bar = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """

        # v_bar = (u_bar  w.T) /  ||u_bar  w.T||
        v_ = tf.matmul(u_bar, tf.transpose(w))
        v_bar = l2_norm(v_)

        # u_bar = (v_bar  w.T) /  ||v_bar  w.T||
        u_ = tf.matmul(v_bar, w)
        u_bar = l2_norm(u_)

    # sigma = v_bar w u_bar.T
    sigma = tf.matmul(tf.matmul(v_bar, w), tf.transpose(u_bar))
    w_sn = w / sigma

    with tf.control_dependencies([u_l.assign(u_bar)]):
        w_sn = tf.reshape(w_sn, w_shape)

    return w_sn

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def hw_flatten(input_) :
    # return tf.reshape(input_, shape=[input_.shape[0], -1, input_.shape[-1]])
    return tf.reshape(input_, shape=(tf.shape(input_)[0], -1, tf.shape(input_)[-1]))
