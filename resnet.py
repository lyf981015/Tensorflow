#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Copyright 2018 AI Futurelab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
# AI FUTURELAB
Image scene classification baseline for regional contest 2018

## Useage

# Training

```
python image_scene_classification.py --mode=train --dataset_dir=<dir> --max_steps=<max_steps> --checkpoint_dir=<checkpoint_dir>
```

 --mode Define the running mode
 --dataset_dir Path to directory of training set, e.g. /home/ubuntu/image_scene_training_v1
 --checkpoint_dir Path to directory of checkpoint files.
 --max_steps Maximum traning steps

# Test
```
python scene.py --mode=test --dataset_dir=<testset_dir> --checkpoint_dir=<checkpoint_dir> --target_file=<target_file>
```

 --mode Define the running mode
 --dataset_dir Path to directory of test set, e.g. /home/ubuntu/image_scene_test_v1
 --checkpoint_dir Path to directory of checkpoint files.
 --target_file Path to result file

'''

import tensorflow as tf
import numpy as np
import time, argparse
import six
from tensorflow.python.training import moving_averages

# local packages
import HEU_nput

LEARNINGRATE = 1e-3
IMAGE_SIDE_LENGTH = 112
IMAGE_CHANNELS = 3
BATCH_SIZE = 20
TOP_N = 3
CATEGORIES = 20
extra_train_ops = []
num_residual_units=5
use_bottleneck=False
weight_decay_rate=0.0002
relu_leakiness=0.1
optimizer='mom'
test_dir ='/home/long/baseline/py'
def train(max_step, checkpoint_dir):   //主训练函数
    # train the model
    scene_data = HEU_nput.TraningSet()
    features = tf.placeholder(np.float32, shape=[None, IMAGE_SIDE_LENGTH, IMAGE_SIDE_LENGTH, IMAGE_CHANNELS], name="features")
    labels = tf.placeholder(np.float32, [None], name="labels") //tensor的引用
    one_hot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=CATEGORIES)
    cross_entropy,logits,train_step= build_model(features,one_hot_labels)
    keep_prob = tf.placeholder(tf.float32)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    with tf.Session() as sess:
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            print('Restore the model from checkpoint %s' % checkpoint.model_checkpoint_path)
            # Restores from checkpoint
            saver.restore(sess, checkpoint.model_checkpoint_path)
            start_step = int(checkpoint.model_checkpoint_path.split('/')[-1].split('-')[-1])
        else:
            sess.run(tf.global_variables_initializer())
            start_step = 0
            print('New training')

        start_time = time.time()
        for step in range(start_step, start_step + max_step):
            x, y = scene_data.next_batch(BATCH_SIZE, IMAGE_SIDE_LENGTH)
            sess.run(train_step, feed_dict={features: x, labels: y, keep_prob: 0.5})
            if step % 1 == 0 and step > 0:
                train_accuracy = sess.run(accuracy, feed_dict={features: x, labels: y, keep_prob: 1})
                train_loss = sess.run(cross_entropy, feed_dict={features: x, labels: y, keep_prob: 1})
                duration = time.time() - start_time
                print ("Step %d: training accuracy %g, loss is %g (%0.3f sec)" % (step, train_accuracy, train_loss, duration))
                start_time = time.time()
            if step % 20 == 0 and step > 0:
                saver.save(sess, checkpoint_dir, global_step=step)
                print('Write checkpoint at %s' % step)

def test(checkpoint_dir, target_file):
    #test_images = os.listdir(test_dir)
    listfile = test_dir+"/list.csv"
    test_list = HEU_nput.read_csvlist(listfile)
    features = tf.placeholder(tf.float32, shape=[None, IMAGE_SIDE_LENGTH, IMAGE_SIDE_LENGTH, IMAGE_CHANNELS], name="features")
    labels = tf.placeholder(tf.float32, [None], name="labels")
    one_hot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=CATEGORIES)
    cross_entropy, logits, train_step = build_model(features, one_hot_labels)
    keep_prob = tf.placeholder(tf.float32)
    values, indices = tf.nn.top_k(logits, TOP_N)

    with tf.Session() as sess:
        saver = tf.train.Saver()

        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        if not checkpoint or not checkpoint.model_checkpoint_path:
            raise Exception('Fail to load checkpoint')

        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Load checkpoint %s" % checkpoint.model_checkpoint_path)

        results = []
        for row in test_list:
            file_id = row[0] #the first cell of a row is file_id
            imagefile = test_dir+'/data/'+file_id+'.jpg'
            x = HEU_nput.img_resize(imagefile, IMAGE_SIDE_LENGTH)
            feed_dict = {features: np.expand_dims(x, axis=0), keep_prob: 1}
            y = np.squeeze(sess.run(indices, feed_dict = feed_dict), axis=0)
            result = [file_id]
            result.extend(y.tolist())
            results.append(result)
            print('FileID: %s, Categroies: %d,%d,%d' % (file_id, y[0], y[1], y[2]))

        HEU_nput.write_csvlist(target_file, results, header=['FILE_ID', 'CATEGORY_ID0', 'CATEGORY_ID1', 'CATEGORY_ID2'])
        print("Write results to file %s" % target_file)
//神经网络主体
def build_model(images,labels):
    with tf.variable_scope('init'):
        x =images
        """第一层卷积（3,3x3/1,16）"""
        x =conv('init_conv',x, 7, 3, 16,stride_arr(1)) //这里应该加一层池化层

    # 残差网络参数
    strides = [1, 2, 2]
    # 激活前置
    activate_before_residual = [True, False, False]
    # bottleneck残差单元模块
    res_func =bottleneck_residual
    # 通道数量
    filters = [16, 64, 128, 256]

    # 第一组
    with tf.variable_scope('unit_1_0'):
        x = res_func(x, filters[0], filters[1],
                        stride_arr(strides[0]),
                        activate_before_residual[0])
    for i in six.moves.range(1,num_residual_units):
        with tf.variable_scope('unit_1_%d'%i):
            x = res_func(x, filters[1], filters[1], stride_arr(1), False)

    # 第二组
    with tf.variable_scope('unit_2_0'):
        x = res_func(x, filters[1], filters[2],
                       stride_arr(strides[1]),
                        activate_before_residual[1])
    for i in six.moves.range(1, num_residual_units):
        with tf.variable_scope('unit_2_%d'%i):
            x = res_func(x, filters[2], filters[2], stride_arr(1), False)

    # 第三组
    with tf.variable_scope('unit_3_0'):
        x = res_func(x, filters[2], filters[3], stride_arr(strides[2]),
                         activate_before_residual[2])
    for i in six.moves.range(1, num_residual_units):
        with tf.variable_scope('unit_3_%d'%i):
            x = res_func(x, filters[3], filters[3], stride_arr(1), False)

    # 全局池化层
    with tf.variable_scope('unit_last'):
        x = batch_norm('final_bn',x,'train')
        x = relu(x,relu_leakiness)
        x = global_avg_pool(x)

    # 全连接层
    with tf.variable_scope('logits'):
        logits = fully_connected(x,CATEGORIES)

    # 构建损失函数
    with tf.variable_scope('costs'):
        # 交叉熵
        xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        # 加和
        cost = tf.reduce_mean(xent,name= 'xent') //隐藏层共44层
        train_step = tf.train.AdamOptimizer(LEARNINGRATE).minimize(cost)
    return cost,logits,train_step
// bottlenck_residual残差模块
def bottleneck_residual(x, in_filter, out_filter, stride,activate_before_residual=False):
    #前置激活
    if activate_before_residual:
        with tf.variable_scope('common_bn_relu'):
            # 先做BN和ReLU激活
            x = batch_norm('init_bn',x,'train')
            x = relu(x,relu_leakiness)
            # 获取残差直连
            orig_x = x
    else:
        with tf.variable_scope('residual-bn_relu'):
            # 获取残差直连
            orig_x = x
            # 后做BN和ReLU激活
            x = batch_norm('init_bn',x,'train')
            x = relu(x,relu_leakiness)
        # 第1子层
    with tf.variable_scope('sub1'):
        # 1x1卷积，使用输入步长，通道数(in_filter -> out_filter/4)
        x = conv('conv1',x, 1, in_filter, out_filter / 4, stride)

    # 第2子层
    with tf.variable_scope('sub2'):
        # BN和ReLU激活
        x =batch_norm('bn2',x,'train')
        x =relu(x,relu_leakiness)
        # 3x3卷积，步长为1，通道数不变(out_filter/4)
        x = conv('conv2',x, 3, out_filter / 4, out_filter / 4, [1, 1, 1, 1])

        # 第3子层
    with tf.variable_scope('sub2'):
        # BN和ReLU激活
        x = batch_norm('bn3',x,'train')
        x = relu(x,relu_leakiness)
        # 1x1卷积，步长为1，通道数不变(out_filter/4 -> out_filter)
        x = conv('conv3',x, 1, out_filter / 4, out_filter, [1, 1, 1, 1])

    # 合并残差层
    with tf.variable_scope('sub_add'):
        # 当通道数有变化时
        if in_filter != out_filter:
            # 1x1卷积，使用输入步长，通道数(in_filter -> out_filter)
             orig_x = conv('project',orig_x, 1, in_filter, out_filter, stride)

            # 合并残差
        x += orig_x
    return x
def stride_arr(stride):
    return [1, stride, stride, 1]
    # 2D卷积
def conv(name,x, filter_size, in_filters, out_filters, strides):
    with tf.variable_scope(name):
        n = filter_size * filter_size * out_filters
        # 获取或新建卷积核，正态随机初始化
        kernel = tf.get_variable(
            'DW',
            [filter_size, filter_size, in_filters, out_filters],
            tf.float32,
            initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
        # 计算卷积
        return tf.nn.conv2d(x, kernel, strides, padding='SAME')
# leaky ReLU激活函数，泄漏参数leakiness为0就是标准ReLU
def relu(x, leakiness=0.0):
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
def batch_norm(name,x,mode):
    with tf.variable_scope(name):
        # 输入通道维数
        params_shape = [x.get_shape()[-1]]
        # offset
        beta = tf.get_variable('beta',
                                params_shape,
                                tf.float32,
                                initializer=tf.constant_initializer(0.0, tf.float32))
        # scale
        gamma = tf.get_variable('gamma',
                                params_shape,
                                tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))

        if mode == 'train':
            # 为每个通道计算均值、标准差
            mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
            # 新建或建立测试阶段使用的batch均值、标准差
            moving_mean = tf.get_variable('moving_mean',
                                            params_shape, tf.float32,
                                            initializer=tf.constant_initializer(0.0, tf.float32),
                                            trainable=False)
            moving_variance = tf.get_variable('moving_variance',
                                                params_shape, tf.float32,
                                                initializer=tf.constant_initializer(1.0, tf.float32),
                                                trainable=False)
            # 添加batch均值和标准差的更新操作(滑动平均)
            # moving_mean = moving_mean * decay + mean * (1 - decay)
            # moving_variance = moving_variance * decay + variance * (1 - decay)
            extra_train_ops.append(moving_averages.assign_moving_average(
                moving_mean, mean, 0.9))
            extra_train_ops.append(moving_averages.assign_moving_average(
                moving_variance, variance, 0.9))
        else:
            # 获取训练中积累的batch均值、标准差
            mean = tf.get_variable('moving_mean',
                                    params_shape, tf.float32,
                                    initializer=tf.constant_initializer(0.0, tf.float32),
                                    trainable=False)
            variance = tf.get_variable('moving_variance',
                                        params_shape, tf.float32,
                                        initializer=tf.constant_initializer(1.0, tf.float32),
                                        trainable=False)
                # 添加到直方图总结
            tf.summary.histogram(mean.op.name, mean)
            tf.summary.histogram(variance.op.name, variance)

        # BN层：((x-mean)/var)*gamma+beta
        y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
        y.set_shape(x.get_shape())
        return y
def fully_connected(x, out_dim):
    # 输入转换成2D tensor，尺寸为[N,-1]
    x = tf.reshape(x, [BATCH_SIZE, 256])
    # 参数w，平均随机初始化，[-sqrt(3/dim), sqrt(3/dim)]*factor
    w = tf.get_variable('DW', [256, out_dim],
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    # 参数b，0值初始化
    b = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer())
    # 计算x*w+b
    return tf.nn.xw_plus_b(x, w, b)
def global_avg_pool(x):
    assert x.get_shape().ndims == 4
    # 在第2&3维度上计算均值，尺寸由WxH收缩为1x1
    return tf.reduce_mean(x, [1, 2])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()    //定义了一些全局量
    parser.add_argument('--mode', type=str, default='test', help="Define the running mode as 'training' or 'test'.")
    parser.add_argument('--checkpoint_dir', type=str, default='/home/long/image_scene_training/', help="Path to directory of checkpoint.")
    parser.add_argument('--max_steps', type=int, default=65000, help="Maximum training steps.")
    parser.add_argument('--target_file', type=str, default='/home/long/baseline/py/test_results.csv', help='Path to test result file.')

    args = parser.parse_args()
    if args.mode == 'train':
        train(args.max_steps, args.checkpoint_dir)
    elif args.mode == 'test':
        test(args.checkpoint_dir, args.target_file)
    else:
        raise Exception('--mode can be train or test only')
