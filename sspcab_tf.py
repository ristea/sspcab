# This code is released under the CC BY-SA 4.0 license.

import tensorflow as tf


# SSPCAB implementation
def sspcab_layer(input, name, kernel_dim, dilation, filters, reduction_ratio=8):
    '''
        input: The input data
        name: The name of the layer in the graph
        kernel_dim: The dimension of the sub-kernels ' k' ' from the paper
        dilation: The dilation dimension 'd' from the paper
        filters: The number of filter at the output (usually the same with the number of filter from the input)
        reduction_ratio: The reduction ratio for the SE block ('r' from the paper)
    '''
    with tf.variable_scope('SSPCAB/' + name) as scope:
        pad = kernel_dim + dilation
        border_input = kernel_dim + 2*dilation + 1
        
        sspcab_input = tf.pad(input, tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]]), "REFLECT")
        
        sspcab_1 = tf.layers.conv2d(inputs=sspcab_input[:, :-border_input, :-border_input, :],
                                    filters=filters, kernel_size=kernel_dim, activation=tf.nn.relu)
        sspcab_3 = tf.layers.conv2d(inputs=sspcab_input[:, border_input:, :-border_input, :],
                                    filters=filters, kernel_size=kernel_dim, activation=tf.nn.relu)
        sspcab_7 = tf.layers.conv2d(inputs=sspcab_input[:, :-border_input, border_input:, :],
                                    filters=filters, kernel_size=kernel_dim, activation=tf.nn.relu)
        sspcab_9 = tf.layers.conv2d(inputs=sspcab_input[:, border_input:, border_input:, :],
                                    filters=filters, kernel_size=kernel_dim, activation=tf.nn.relu)
        sspcab_out = sspcab_1 + sspcab_3 + sspcab_7 + sspcab_9

        se_out = se_layer(sspcab_out, filters, reduction_ratio, 'SSPCAB/se_' + name)
        return se_out


# Squeeze and Excitation block
def se_layer(input_x, in_channels, ratio, layer_name):
    '''
        input_x: The input data
        out_dim: The number of input channels
        ration: The reduction ratio 'r' from the paper
        layer_name: The name of the layer in the graph
    '''
    with tf.name_scope(layer_name):
        squeeze = tf.reduce_mean(input_x, axis=[1, 2])
        excitation = tf.layers.dense(squeeze, use_bias=True, units=in_channels / ratio)
        excitation = tf.nn.relu(excitation)
        excitation = tf.layers.dense(excitation, use_bias=True, units=in_channels)
        excitation = tf.nn.sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1, 1, 1, in_channels])
        scale = input_x * excitation
        return scale

# Example of how our block should be updated
# cost_sspcab = tf.square(self.input_sspcab - output_sspcab)
# loss = 0.1 * tf.reduce_mean(cost_sspcab)
