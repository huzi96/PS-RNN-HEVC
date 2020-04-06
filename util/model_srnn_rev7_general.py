import tensorflow as tf
import numpy as np
# refined srnn based on rev8
# try stack CUDNN GRU

def stacked_RNN(input_tensor, num, scope, units, batch_size, num_scale):
    with tf.variable_scope(scope):
        cells = [tf.contrib.rnn.GRUCell(num_units=units,name='cell_%d'% (i) )for i in range(num)]
        states = [it.zero_state(batch_size, dtype=tf.float32) for i,it in enumerate(cells)]
        last = input_tensor
        for i in range(num):
            last, _ = tf.nn.static_rnn(cells[i], last, initial_state=states[i], sequence_length=[num_scale,]*batch_size, scope='rnn_%d' % (i))
        return last

def inter_rnn(input_tensor, num, scope, batch_size, channels=8, units=8):
    with tf.variable_scope(scope):
        # split input tensor to lines
        shaped_conv1 = tf.reshape(input_tensor, [batch_size, num, num, channels], name="conv1")

        vertical_form = tf.reshape(shaped_conv1, [batch_size, num, num*channels], name='vertical_form')
        horizontal_form1 = tf.transpose(shaped_conv1, [0, 2, 1, 3], name='horizontal_form1')
        horizontal_form =tf.reshape(horizontal_form1, [batch_size, num, num*channels], name='horizontal_form')

        vertical_split = tf.unstack(
            vertical_form,
            num=num,
            axis=1,
            name="vertical_split"
        )

        horizontal_split = tf.unstack(
            horizontal_form,
            num=num,
            axis=1,
            name="horizontal_split"
        )

        vr4 = stacked_RNN(vertical_split, 1, 'vrnn', num*channels, batch_size, num)
        hr4 = stacked_RNN(horizontal_split, 1, 'hrnn', num*channels, batch_size, num)

        stack_h_ = tf.stack(hr4, axis=1, name='from_h')

        stack_v_ = tf.stack(vr4, axis=1, name='from_v')

        _stack_h = tf.reshape(stack_h_, [batch_size, num, num, channels], name='stack_shape_h')
        stack_v = tf.reshape(stack_v_, [batch_size, num, num, channels], name='stack_shape_v')

        stack_h = tf.transpose(_stack_h, [0,2,1,3], name='h_stack_back')

        concat2 = tf.concat([stack_v, stack_h], axis=3)

        _connect = tf.layers.conv2d(
            inputs=concat2,
            filters=units,
            kernel_size=[1, 1],
            strides=[1,1],
            padding="VALID",
            name="connect"
        )

        connect = tf.keras.layers.PReLU(shared_axes=[1,2], name='relu_con')(_connect)


        return connect


def build_model(input_tensor, target_tensor, params, mode=3):

    print("mode : %d" % (mode))
    print(input_tensor.shape)
    batch_size = params['batch_size']

    num_scale = params['num_scale']

    input_layer = tf.reshape(input_tensor, [-1, num_scale*mode, num_scale*mode, 1])

    _convdown = tf.layers.conv2d(
        inputs=input_layer,
        filters=16,
        kernel_size=[1, 1],
        strides=[1,1],
        padding="SAME",
        name="convdown"
    )

    convdown = tf.keras.layers.PReLU(shared_axes=[1,2], name='reludown')(_convdown)

    _conv1 = tf.layers.conv2d(
        inputs=convdown,
        filters=16,
        kernel_size=[3, 3],
        padding='SAME',
        name="conv1"
    )

    conv1 = tf.keras.layers.PReLU(shared_axes=[1,2])(_conv1)

    rnn1 = inter_rnn(conv1, num_scale*mode, 'inter1', batch_size, 16, 8)
    print(rnn1.shape)

    # rnn2 = inter_rnn(rnn1, num_scale*mode, 'inter2', batch_size, 16, 16)
    # print(rnn2.shape)

    _convdown1 = tf.layers.conv2d(
        inputs=rnn1,
        filters=8,
        kernel_size=[mode*2+1, mode*2+1],
        strides=[mode,mode],
        padding="SAME",
        name="convdown1"
    )

    convdown1 = tf.keras.layers.PReLU(shared_axes=[1,2], name='reludown1')(_convdown1)

    rnn3 = inter_rnn(convdown1, num_scale, 'inter3', batch_size, 8, 8)
    print(rnn3.shape)

    rnn4 = inter_rnn(rnn3, num_scale, 'inter4', batch_size, 8, 8)
    print(rnn4.shape)

    _conv10 = tf.layers.conv2d(
        rnn4,
        filters=16,
        kernel_size=[3, 3],
        padding="SAME",
        name='conv10'
    )

    conv10 = tf.keras.layers.PReLU(shared_axes=[1,2], name='relu10')(_conv10)

    conv11 = tf.layers.conv2d(
        conv10,
        filters=1,
        kernel_size=[1, 1],
        padding="SAME", name='conv11'
    )

    def SATD(y_true, y_pred, scale):
        H_8x8 = np.array(
            [[1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
             [1., -1.,  1., -1.,  1., -1.,  1., -1.],
             [1.,  1., -1., -1.,  1.,  1., -1., -1.],
             [1., -1., -1.,  1.,  1., -1., -1.,  1.],
             [1.,  1.,  1.,  1., -1., -1., -1., -1.],
             [1., -1.,  1., -1., -1.,  1., -1.,  1.],
             [1.,  1., -1., -1., -1., -1.,  1.,  1.],
             [1., -1., -1.,  1., -1.,  1.,  1., -1.]],
            dtype=np.float32
        )
        H_target = np.zeros((1, 32,32), dtype=np.float32)
        H_target[0, 0:8,0:8] = H_8x8

        H_target[0, 0:8,8:16] = H_8x8
        H_target[0, 8:16,0:8] = H_8x8
        H_target[0, 8:16,8:16] = -H_8x8

        H_target[0, 16:32, 0:16] = H_target[0, 0:16, 0:16]
        H_target[0, 0:16, 16:32] = H_target[0, 0:16, 0:16]
        H_target[0, 16:32, 16:32] = -H_target[0, 0:16, 0:16]

        TH0 = tf.constant(H_target[:, 0:scale, 0:scale])

        TH1 = tf.tile(TH0, (batch_size, 1, 1))

        diff = tf.reshape(y_true - y_pred, (-1, scale, scale))

        return tf.reduce_mean(tf.square(tf.matmul(tf.matmul(TH1, diff), TH1)))

    loss = SATD(target_tensor, conv11, num_scale)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(params['learning_rate'], global_step=global_step, decay_steps = 160000, decay_rate=0.1)
    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    mse_loss = tf.reduce_mean(tf.square((target_tensor-conv11)))

    return train_op, loss, mse_loss
