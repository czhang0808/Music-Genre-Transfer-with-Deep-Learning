from __future__ import division
import tensorflow as tf
from ops import *
from utils import *


def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


def softmax_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))


def discriminator(image, options, reuse=False, name="discriminator", attention=True):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # h0 = lrelu(conv2d(image, options.df_dim, ks=[12, 12], s=[12, 12], name='d_h0_conv'))
        # h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*4, ks=[4, 1], s=[4, 1], name='d_h1_conv'), 'd_bn1'))
        # h4 = conv2d(h1, 1, s=1, name='d_h3_pred')

        # # input is (64 x 84 x self.df_dim)
        # h0 = lrelu(conv2d(image, options.df_dim, ks=[1, 12], s=[1, 12], name='d_h0_conv'))
        # # h0 is (64 x 7 x self.df_dim)
        # h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, ks=[2, 1], s=[2, 1], name='d_h1_conv'), 'd_bn1'))
        # # h1 is (32 x 7 x self.df_dim*2)
        # h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, ks=[2, 1], s=[2, 1], name='d_h2_conv'), 'd_bn2'))
        # # h2 is (16x 7 x self.df_dim*4)
        # h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, ks=[2, 1], s=[2, 1], name='d_h3_conv'), 'd_bn3'))
        # # h3 is (8 x 7 x self.df_dim*8)
        # h4 = conv2d(h3, 1, s=1, name='d_h3_pred')
        # # h4 is (8 x 7 x 1)

        h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
        # (32, 42, 64)
        if attention:
        # attention layer
            out1, att1 = self_attention(in_tensor=h0, in_dim=options.df_dim, name='d_att1')
            h1 = lrelu(instance_norm(conv2d(out1, options.df_dim * 4, name='d_h1_conv'), 'd_bn1'))
            out2, att2 = self_attention(in_tensor=h1, in_dim=options.df_dim * 4, name='d_att2')
            h2 = conv2d(out2, 1, s=1, name='d_h2_pred')
        else:
            h1 = lrelu(instance_norm(conv2d(h0, options.df_dim * 4, name='d_h1_conv'), 'd_bn1'))
            # (16, 21, 256)
            h2 = conv2d(h1, 1, s=1, name='d_h2_pred')
            # (16, 21, 1)
        return h2




    dropout_rate = 0.5 if options.is_training else 1.0
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # image is (256 x 256 x input_c_dim)
        e1 = instance_norm(conv2d(image, options.gf_dim, name='g_e1_conv'))
        # e1 is (128 x 128 x self.gf_dim)
        e2 = instance_norm(conv2d(lrelu(e1), options.gf_dim*2, name='g_e2_conv'), 'g_bn_e2')
        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = instance_norm(conv2d(lrelu(e2), options.gf_dim*4, s=3, name='g_e3_conv'), 'g_bn_e3')
        # e3 is (32 x 32 x self.gf_dim*4)
        e4 = instance_norm(conv2d(lrelu(e3), options.gf_dim*8, s=[2, 1], name='g_e4_conv'), 'g_bn_e4')
        # e4 is (16 x 16 x self.gf_dim*8)
        e5 = instance_norm(conv2d(lrelu(e4), options.gf_dim*8, s=[2, 1], name='g_e5_conv'), 'g_bn_e5')
        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = instance_norm(conv2d(lrelu(e5), options.gf_dim*8, s=[2, 7], name='g_e6_conv'), 'g_bn_e6')
        # e6 is (4 x 4 x self.gf_dim*8)
        e7 = instance_norm(conv2d(lrelu(e6), options.gf_dim*8, s=[2, 1], name='g_e7_conv'), 'g_bn_e7')
        # e7 is (2 x 2 x self.gf_dim*8)
        e8 = instance_norm(conv2d(lrelu(e7), options.gf_dim*8, s=[2, 1], name='g_e8_conv'), 'g_bn_e8')
        # e8 is (1 x 1 x self.gf_dim*8)

        d1 = deconv2d(tf.nn.relu(e8), options.gf_dim*8, s=[2, 1], name='g_d1')
        d1 = tf.nn.dropout(d1, dropout_rate)
        d1 = tf.concat([instance_norm(d1, 'g_bn_d1'), e7], 3)
        # d1 is (2 x 2 x self.gf_dim*8*2)

        d2 = deconv2d(tf.nn.relu(d1), options.gf_dim*8, s=[2, 1], name='g_d2')
        d2 = tf.nn.dropout(d2, dropout_rate)
        d2 = tf.concat([instance_norm(d2, 'g_bn_d2'), e6], 3)
        # d2 is (4 x 4 x self.gf_dim*8*2)

        d3 = deconv2d(tf.nn.relu(d2), options.gf_dim*8, s=[2, 7], name='g_d3')
        d3 = tf.nn.dropout(d3, dropout_rate)
        d3 = tf.concat([instance_norm(d3, 'g_bn_d3'), e5], 3)
        # d3 is (8 x 8 x self.gf_dim*8*2)

        d4 = deconv2d(tf.nn.relu(d3), options.gf_dim*8, s=[2, 1], name='g_d4')
        d4 = tf.concat([instance_norm(d4, 'g_bn_d4'), e4], 3)
        # d4 is (16 x 16 x self.gf_dim*8*2)

        d5 = deconv2d(tf.nn.relu(d4), options.gf_dim*4, s=[2, 1], name='g_d5')
        d5 = tf.concat([instance_norm(d5, 'g_bn_d5'), e3], 3)
        # d5 is (32 x 32 x self.gf_dim*4*2)

        d6 = deconv2d(tf.nn.relu(d5), options.gf_dim*2, s=3, name='g_d6')
        d6 = tf.concat([instance_norm(d6, 'g_bn_d6'), e2], 3)
        # d6 is (64 x 64 x self.gf_dim*2*2)

        d7 = deconv2d(tf.nn.relu(d6), options.gf_dim, name='g_d7')
        d7 = tf.concat([instance_norm(d7, 'g_bn_d7'), e1], 3)
        # d7 is (128 x 128 x self.gf_dim*1*2)

        d8 = deconv2d(tf.nn.relu(d7), options.output_c_dim, name='g_d8')
        # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(d8)


def generator_resnet(image, options, reuse=False, name="generator", attention = True):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            # e.g, x is (# of images * 128 * 128 * 3)
            p = int((ks - 1) / 2)
            # For ks = 3, p = 1
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            # After first padding, (# of images * 130 * 130 * 3)
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
            # After first conv2d, (# of images * 128 * 128 * 3)
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            # After second padding, (# of images * 130 * 130 * 3)
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
            # After second conv2d, (# of images * 128 * 128 * 3)
            return relu(y + x)

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3

        # Original image is (# of images * 256 * 256 * 3)
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        # c0 is (# of images * 262 * 262 * 3)
        c1 = relu(instance_norm(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        # c1 is (# of images * 256 * 256 * 64)
        c2 = relu(instance_norm(conv2d(c1, options.gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        # c2 is (# of images * 128 * 128 * 128)
        c3 = relu(instance_norm(conv2d(c2, options.gf_dim*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        # c3 is (# of images * 64 * 64 * 256)

        # c4 = relu(instance_norm(conv2d(c3, options.gf_dim*8, 3, 3, name='g_e4_c'), 'g_e4_bn'))
        # c5 = relu(instance_norm(conv2d(c4, options.gf_dim*16, 3, [4, 1], name='g_e5_c'), 'g_e5_bn'))

        # define G network with 9 resnet blocks
        r1 = residule_block(c3, options.gf_dim*4, name='g_r1')
        # r1 is (# of images * 64 * 64 * 256)
        r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        # r2 is (# of images * 64 * 64 * 256)
        r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        # r3 is (# of images * 64 * 64 * 256)
        r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        # r4 is (# of images * 64 * 64 * 256)
        r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        # r5 is (# of images * 64 * 64 * 256)
        r6 = residule_block(r5, options.gf_dim*4, name='g_r6')
        # r6 is (# of images * 64 * 64 * 256)
        r7 = residule_block(r6, options.gf_dim*4, name='g_r7')
        # r7 is (# of images * 64 * 64 * 256)
        r8 = residule_block(r7, options.gf_dim*4, name='g_r8')
        # r8 is (# of images * 64 * 64 * 256)
        r9 = residule_block(r8, options.gf_dim*4, name='g_r9')
        # r9 is (# of images * 64 * 64 * 256)

        if attention:
        # attention layer
            out1, att1 = self_attention(in_tensor=r9, in_dim=options.gf_dim*4, name='g_att1')
            r10 = residule_block(out1, options.gf_dim*4, name='g_r10')
            out2, att2 = self_attention(in_tensor=r10, in_dim=options.gf_dim*4, name='g_att2')
            d1 = relu(instance_norm(deconv2d(out2, options.gf_dim*2, 3, 2, name='g_d1_dc'), 'g_d1_bn'))
        else:
            r10 = residule_block(r9, options.gf_dim*4, name='g_r10')
            d1 = relu(instance_norm(deconv2d(r10, options.gf_dim*2, 3, 2, name='g_d1_dc'), 'g_d1_bn'))
            # d1 is (# of images * 128 * 128 * 128)

        # d4 = relu(instance_norm(deconv2d(r9, options.gf_dim*8, 3, [4, 1], name='g_d4_dc'), 'g_d4_bn'))
        # d5 = relu(instance_norm(deconv2d(d4, options.gf_dim*4, 3, 3, name='g_d5_dc'), 'g_d5_bn'))

        d2 = relu(instance_norm(deconv2d(d1, options.gf_dim, 3, 2, name='g_d2_dc'), 'g_d2_bn'))
        # d2 is (# of images * 256 * 256 * 64)
        d3 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        # After padding, (# of images * 262 * 262 * 64)
        pred = tf.nn.sigmoid(conv2d(d3, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c'))
        # Output image is (# of images * 256 * 256 * 3)

        return pred


def discriminator_classifier(image, options, reuse=False, name="discriminator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        # # input is 384, 84, 1
        # h0 = lrelu(conv2d(image, options.df_dim, [12, 12], [12, 12], name='d_h0_conv'))
        # # h0 is (32 x 7 x self.df_dim)
        # h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, [2, 1], [2, 1], name='d_h1_conv'), 'd_bn1'))
        # # h1 is (16 x 7 x self.df_dim*2)
        # h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, [2, 1], [2, 1], name='d_h2_conv'), 'd_bn2'))
        # # h2 is (8 x 7 x self.df_dim*4)
        # h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, [8, 1], [8, 1], name='d_h3_conv'), 'd_bn3'))
        # # h3 is (1 x 7 x self.df_dim*8)
        # h4 = conv2d(h3, 2, [1, 7], [1, 7], name='d_h3_pred')
        # # h4 is (1 x 1 x 2)

        # input is 64, 84, 1
        h0 = lrelu(conv2d(image, options.df_dim, [1, 12], [1, 12], name='d_h0_conv', sn=False))
        # h0 is (64 x 7 x self.df_dim)
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim * 2, [4, 1], [4, 1], name='d_h1_conv', sn=False), 'd_bn1'))
        # h1 is (16 x 7 x self.df_dim*2)
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim * 4, [2, 1], [2, 1], name='d_h2_conv', sn=False), 'd_bn2'))
        # h2 is (8 x 7 x self.df_dim*4)
        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim * 8, [8, 1], [8, 1], name='d_h3_conv', sn=False), 'd_bn3'))
        # h3 is (1 x 7 x self.df_dim*8)
        h4 = conv2d(h3, 2, [1, 7], [1, 7], name='d_h3_pred', sn=False)
        # h4 is (1 x 1 x 2)
        return tf.reshape(h4, [-1, 2])  # batch_size * 2


def discriminator_classifier_new(image, options, reuse=False, name="discriminator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        # # input is 384, 84, 1
        # h0 = lrelu(conv2d(image, options.df_dim, [12, 12], [12, 12], name='d_h0_conv'))
        # # h0 is (32 x 7 x self.df_dim)
        # h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, [2, 1], [2, 1], name='d_h1_conv'), 'd_bn1'))
        # # h1 is (16 x 7 x self.df_dim*2)
        # h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, [2, 1], [2, 1], name='d_h2_conv'), 'd_bn2'))
        # # h2 is (8 x 7 x self.df_dim*4)
        # h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, [8, 1], [8, 1], name='d_h3_conv'), 'd_bn3'))
        # # h3 is (1 x 7 x self.df_dim*8)
        # h4 = conv2d(h3, 2, [1, 7], [1, 7], name='d_h3_pred')
        # # h4 is (1 x 1 x 2)

        # input is 64, 84, 1
        h0 = lrelu(conv2d(image, options.df_dim, [1, 12], [1, 12], name='d_h0_conv', sn=False))
        # h0 is (64 x 7 x self.df_dim)
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim * 2, [4, 1], [4, 1], name='d_h1_conv', sn=False), 'd_bn1'))
        # h1 is (16 x 7 x self.df_dim*2)        
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim * 4, [2, 1], [2, 1], name='d_h2_conv', sn=False), 'd_bn2'))
        # h2 is (8 x 7 x self.df_dim*4)
        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim * 8, [8, 1], [8, 1], name='d_h3_conv', sn=False), 'd_bn3'))
        # h3 is (1 x 7 x self.df_dim*8)
        h4 = conv2d(h3, 3, [1, 7], [1, 7], name='d_h4_pred', sn=False)
        # h4 is (1 x 1 x 3)
        # h0 = lrelu(conv2d(image, options.df_dim * 2, [1, 4], [1, 4], name='d_h0_conv', sn=False))
        # # h0 is (64 x 21 x self.df_dim)
        # h1 = lrelu(instance_norm(conv2d(h0, options.df_dim * 4, [4, 1], [4, 1], name='d_h1_conv'), 'd_bn1'))
        # # h1 is (16 x 21 x self.df_dim*4)
        # h2 = lrelu(instance_norm(conv2d(h1, options.df_dim * 8, [2, 3], [2, 3], name='d_h2_conv'), 'd_bn2'))
        # # h1 is (8 x 7 x self.df_dim*8)           
        # h3 = lrelu(instance_norm(conv2d(h2, options.df_dim * 8, [2, 1], [2, 1], name='d_h3_conv'), 'd_bn3'))
        # # h3 is (4 x 7 x self.df_dim*8)
        # h4 = lrelu(instance_norm(conv2d(h3, options.df_dim * 16, [1, 1], [1, 1], name='d_h4_conv'), 'd_bn4'))
        # # h3 is (4 x 7 x self.df_dim*16)
        # h5 = conv2d(h4, 3, [4, 7], [4, 7], name='d_h5_pred')
        # # h5 is (1 x 1 x 3)
        # return tf.reshape(h5, [-1, 3])  # batch_size * 3        
        return tf.reshape(h4, [-1, 3])  # batch_size * 3




    with tf.variable_scope(name, reuse=reuse):

        ## conv
        h0 = lrelu(conv2d_musegan(tf.expand_dims(in_tensor, axis=2), 512, kernels=[2, 1], strides=[1, 1], name='h0'))
        h1 = lrelu(conv2d_musegan(h0, 128, kernels=[3, 1], strides=[3, 1], name='h1'))

        ## linear
        h2 = tf.reshape(h1, [-1, np.product([s.value for s in h1.get_shape()[1:]])])
        h2 = lrelu(linear(h2, 1024, scope='h2'))
        h3 = linear(h2, 1, scope='h3')

    return h3

def self_attention(in_tensor, in_dim, name = 'self_attention'):
    # with tf.variable_scope(name):
    #     if reuse:
    #         tf.get_variable_scope().reuse_variables()
    #     else:
    #         assert tf.get_variable_scope().reuse is False
        # output size = (bs, h , w, channels)
        f = conv2d(input_ = in_tensor, output_dim = in_dim//8, ks = 1, s = 1, padding = 'VALID', name = name+'_f')
        g = conv2d(input_ = in_tensor, output_dim = in_dim//8, ks = 1, s = 1, padding = 'VALID', name = name+'_g')
        h = conv2d(input_ = in_tensor, output_dim = in_dim, ks = 1, s = 1, padding = 'VALID', name = name+'_h')

        #output size = (bs, N, N), N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b = True)
        attention = tf.nn.softmax(s)

        #output size = # (bs, N, in_dim)
        o = tf.matmul(attention, hw_flatten(h))
        gamma = tf.get_variable(name=name+"_gamma", shape=[1], initializer=tf.constant_initializer(0.0))
        #output size  = (bs, h, w, in_dim)
        o = tf.reshape(o, shape = tf.shape(in_tensor))
        out = gamma * o + in_tensor

        return out, attention

