#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
from PIL import Image
import multiprocessing
from copy import deepcopy
from skimage import transform
import cv2
import pydicom

str_result = ['static-state', 'figer', 'lips', 'press', 'emotion', 'language',
       'gambling', 'motor', 'wm', 'social', 'relational']
# In[2]:

d = '/var/www/drag_uploads/fMRI/'
dataPath = d + '/data/'
out_path = d + '/images/'
model_path = d + '/model/'
model_name = 'Model__Step_18000_8_0.0003'
retval = 2000


# In[3]:


def read_data(dataPath):
    img = np.zeros((31, 128, 128))
    for i in range(31):
        temp = pydicom.dcmread(dataPath + '{:d}'.format(i) + '.dcm').pixel_array.astype(np.float32)
        temp = cv2.GaussianBlur(temp, (3, 3), 0)
        ret, temp = cv2.threshold(temp, retval, 1, cv2.THRESH_TOZERO)
        img[i] = temp
    img = img.astype(np.float64)
    img = (img - retval)/10000 - 0.5
    img = np.reshape(img, [1, 31, 128, 128, 1])
    return img


# In[4]:


# s 步长，channels_in 输入通道，channels_out 输出通道
def conv3d_layer(X, k, s, channels_in, channels_out, is_training, name = 'CONV'):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([k, k, k, channels_in, channels_out], stddev = 0.1));
        b = tf.Variable(tf.constant(0.01, shape = [channels_out]))
        conv = tf.nn.conv3d(X, W, strides = [1, s, s, s, 1], padding = 'SAME') + b
#         bn = tf.layers.batch_normalization(conv, training = is_training)
        result = tf.nn.relu(conv)
        tf.summary.histogram('weights', W)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activations', result)
        return result
    
def pool3d_layer(X, k, s, strr = 'SAME', pool_type = 'MAX', name = 'pool', down_stride = 1):
    if pool_type == 'MAX':
        result = tf.nn.max_pool3d(X,
                              ksize = [1, down_stride, k, k, 1],
                              strides = [1, down_stride, s, k, 1],
                              padding = strr,
                              name = name)
    else:
        result = tf.nn.avg_pool3d(X,
                              ksize = [1, down_stride, k, k, 1],
                              strides = [1, down_stride, s, k, 1],
                              padding = strr,
                              name = name)
    return result

def fc_layer(X, neurons_in, neurons_out, last = False, name = 'FC'):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([neurons_in, neurons_out], stddev = 0.1))
        b = tf.Variable(tf.constant(0.01, shape = [neurons_out]))
        tf.summary.histogram('weights', W)
        tf.summary.histogram('biases', b)
        if last == False:
            result = tf.nn.relu(tf.matmul(X, W) + b)
        else:
            result = tf.matmul(X, W) + b
        tf.summary.histogram('activations', result)
        return result


# In[5]:


def Network(BatchSize = 1, learning_rate = 0.001, dataPath = dataPath, out_path = out_path):
    tf.reset_default_graph()
    with tf.Session() as sess:
        is_training = tf.placeholder(dtype = tf.bool, shape=())
        keep_prob = tf.placeholder('float32', name = 'keep_prob')
        
        judge = tf.Print(is_training, ['is_training:', is_training])
    
        
        image_Batch = tf.placeholder('float32', shape = [1, 31, 128, 128, 1], name = 'image_data')
        label_Batch = 0
        
        label_Batch = tf.one_hot(label_Batch, depth = 3)
        


        X = tf.identity(image_Batch)
        y = tf.identity(label_Batch)
        
        conv1_1 = conv3d_layer(X, 3, 1, 1, 16, is_training, "conv1_1")
        conv1_2 = conv3d_layer(conv1_1, 3, 1, 16, 16, is_training, "conv1_2")
#         conv1_3 = conv3d_layer(conv1_2, 3, 1, 16, 16, is_training, "conv1_3")
        pool1 = pool3d_layer(conv1_2, 2, 2, "SAME", "MAX", 'pool1')

        conv2_1 = conv3d_layer(pool1, 3, 1, 16, 32, is_training, 'conv2_1')
#         conv2_2 = conv3d_layer(conv2_1, 3, 1, 32, 32, is_training, 'conv2_2')
        pool2 = pool3d_layer(conv2_1, 2, 2, "SAME", "MAX", 'pool2')
        
        conv3 = conv3d_layer(pool2, 3, 1, 32, 16, is_training, 'conv3')
        pool3 = pool3d_layer(conv3, 2, 2, "SAME", "MAX", 'pool3', down_stride = 2)
#         print(pool3.shape)
        
        drop1 = tf.nn.dropout(pool3, keep_prob)
        fc1 = fc_layer(tf.reshape(drop1, [-1, 16 * 16 * 16 * 16]), 16 * 16 * 16 * 16, 1024, name = 'fc1')
        
        drop2 = tf.nn.dropout(fc1, keep_prob)
        fc2 = fc_layer(drop2, 1024, 128, name = 'fc2')
        
        drop3 = tf.nn.dropout(fc2, keep_prob)
        y_result = fc_layer(drop3, 128, 3, True, name = 'fc3')
        
        
        
#         conv4 = conv3d_layer(pool3, 3, 1, 38, 3, 'conv4')
#         pool4 = pool3d_layer(conv4, 11, 11, "SAME", "MEAN", 'pool4')
#         print(pool4.shape)
        
#         y_result = tf.reshape(pool4, [-1, 3])
#         print(y_result.shape)
        
        with tf.name_scope('summaries'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops): 
    #             cross_entropy = -tf.reduce_mean(y * tf.log(tf.clip_by_value(y_result, 1e-10,1.0)))
                cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_result, labels = y))
                train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
                #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
                corrent_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_result, 1))
                accuracy = tf.reduce_mean(tf.cast(corrent_prediction, 'float', name = 'accuracy'))
                tf.summary.scalar("loss", cross_entropy)
                tf.summary.scalar("accuracy", accuracy)
            
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        
        merge_summary = tf.summary.merge_all()
        summary__train_writer = tf.summary.FileWriter("./logs/train" + '_rate:' + str(learning_rate) + '_' + str(BatchSize), sess.graph)
        summary_val_writer = tf.summary.FileWriter("./logs/test" + '_rate:' + str(learning_rate) + '_' + str(BatchSize))
        
        saver = tf.train.Saver()
        saver.restore(sess, model_path + model_name)
        
        ########################################################################
        signal = tf.multiply(y_result, y)
        signal = tf.reduce_mean(signal)
        gradient_y_image = tf.gradients(signal, pool3)[0]
        gradient_y_image = tf.nn.relu(gradient_y_image)
        gradient_y_image = tf.div(gradient_y_image, tf.reduce_max(gradient_y_image) + tf.constant(1e-5))
#         gradient_y_image = tf.div(gradient_y_image, tf.sqrt(tf.reduce_mean(tf.square(gradient_y_image))) + tf.constant(1e-5))
#       
        guided_gradient = tf.gradients(cross_entropy, X)
        
        
#         T1 = tf.image.resize_images(conv2_3, [28, 28], method = 0)
        T1 = pool3
        w1 = gradient_y_image
        g1 = guided_gradient
        
        prediction = tf.argmax(y_result, 1)
        label = tf.argmax(y_result, 1)
        ########################################################################

        T, w, g, loss, predic, label1, image = sess.run([T1, w1, g1, cross_entropy, prediction, label, X], 
                                                        feed_dict = {keep_prob: 1.0, 
                                                                     is_training: False, 
                                                                     image_Batch: read_data(dataPath)
                                                                    }
                                                       )
#         print(loss, predic[0])
#         print(predic[0])

#             print(T)

        T = np.array(T[0])
        w = np.array(w[0])
        g = np.array(g)

        Tshape = T.shape
        wshape = w.shape

#         print("T:", T.shape)
#         print("w:", w.shape)

        w = w.mean((0, 1, 2))
        w = w.reshape(wshape[3])

#         print("T:", T.shape)
#         print("w:", w.shape)

        heatmap = np.zeros([Tshape[0], Tshape[1], Tshape[2]])

        w = np.maximum(w, 0)
        for i in range(wshape[3]):
            heatmap += w[i] * T[:, :, :, i]
        heatmap = heatmap / (np.max(heatmap) + 1e-5)

        heatmap = transform.resize(heatmap, (31, 128, 128))

        image = image.reshape([31, 128, 128])
#         plt.show()i

#         plt.figure(figsize=(35, 130))

        for i in range(31):
            plt.axis('off')
            fig = plt.gcf()
            fig.set_size_inches(7.0/3,7.0/3) #dpi = 300, output = 700*700 pixels
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            fig.set_size_inches(7, 7)

#             plt.subplot(11, 3, i + 1)
            plt.imshow(image[i])
            plt.imshow(heatmap[i], cmap = 'jet', alpha = 0.5)
            plt.savefig(out_path + '{}.jpg'.format(i))
            plt.close() 
#                 plt.colorbar()

        img = np.zeros([128, 128])
        plt.axis('off')
        fig = plt.gcf()
        fig.set_size_inches(7.0/3,7.0/3) #dpi = 300, output = 700*700 pixels
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        fig.set_size_inches(7, 7)

        plt.imshow(img)
        plt.imshow(img, cmap = 'jet', alpha = 0.5)
        plt.savefig(out_path + '31.jpg')
        plt.cla()

        coord.request_stop()
        coord.join(threads)
        sess.close()
        
        return int(predic[0]) + 1


# In[ ]:


#1, 12, 48, 24, 512, 0.003
def main():
    os.system('rm -f /var/www/html/fmri_images/fMRI/result.re')
    os.system('rm -rf /var/www/html/fmri_images/*')
    os.system('rm -rf /var/www/drag_uploads/fMRI/data//*')
    os.system('unzip -o /var/www/drag_uploads/data.zip -d /var/www/drag_uploads/fMRI/data/')
    os.system('cp /var/www/drag_uploads/fMRI/data/*/* /var/www/drag_uploads/fMRI/data/')
    result = Network(out_path = '/var/www/html/fmri_images/')
    print(result)
    print(str_result[int(result)])
    with open('/var/www/drag_uploads/fMRI/result.re', 'w') as fw:
        fw.write(str_result[int(result)])
    
if __name__ == '__main__':
    main()


# In[ ]:




