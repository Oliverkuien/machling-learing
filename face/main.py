import Image
import numpy as np
import tensorflow as tf
import os
#import sys
def load_data(path):
	img = np.asarray(Image.open(path),dtype='float64') / 256
	faces = np.empty((400,57*47))
	for row in range(20):
	 for column in range(20):
	  faces[20 * row + column] = np.ndarray.flatten(img[row*57:(row+1)*57,column*47:(column+1)*47])
	label = np.zeros((400,40))
	for i in range(40):
	 label[i*10:(i+1)*10,i]=1
	train_data = np.empty((320, 57 * 47))
    	train_label = np.zeros((320, 40))
    	vaild_data = np.empty((40, 57 * 47))
    	vaild_label = np.zeros((40, 40))
    	test_data = np.empty((40, 57 * 47))
    	test_label = np.zeros((40, 40))
	
        for i in range(40):
         train_data[i * 8: i * 8 + 8] = faces[i * 10: i * 10 + 8]
         train_label[i * 8: i * 8 + 8] = label[i * 10: i * 10 + 8]

         vaild_data[i] = faces[i * 10 + 8]
         vaild_label[i] = label[i * 10 + 8]

         test_data[i] = faces[i * 10 + 9]
         test_label[i] = label[i * 10 + 9]
	return [(train_data, train_label), (vaild_data, vaild_label),(test_data, test_label)]
def convolutional_layer(data,kernel_size,bias_size,pooling_size):
	kernel = tf.get_variable('conv',kernel_size,initializer=tf.random_normal_initializer())
	bias =tf.get_variable('bias',bias_size,initializer = tf.random_normal_initializer())
	conv = tf.nn.conv2d(data,kernel,strides=[1,1,1,1],padding='SAME')
	linear_output = tf.nn.relu(tf.add(conv,bias))
	pooling = tf.nn.max_pool(linear_output,ksize=pooling_size,strides=pooling_size,padding='SAME')
	return pooling
def linear_layer(data,weights_size,biases_size):
	weights = tf.get_variable('weights',weights_size,initializer = tf.random_normal_initializer())
	biases = tf.get_variable('biases',biases_size,initializer = tf.random_normal_initializer())
	return tf.add(tf.matmul(data,weights),biases)
def convolutional_neural_network(data):
	kernel_shape1 = [5,5,1,32]
	kernel_shape2= [5,5,32,64]
	full_conn_w_shape = [15*12*64,1024]
	out_w_shape = [1024,40]
	bias_shape1 = [32]
	bias_shape2 = [64]
	full_conn_b_shape = [1024]
	out_b_shape = [40]
	data = tf.reshape(data, [-1,57,47,1])
	with tf.variable_scope("conv_layer1") as layer1:
         layer1_output = convolutional_layer(data=data,kernel_size=kernel_shape1,bias_size=bias_shape1,pooling_size=[1, 2, 2, 1])
    
    	with tf.variable_scope("cocnv_layer2") as layer2:
         layer2_output = convolutional_layer(data=layer1_output,kernel_size=kernel_shape2,bias_size=bias_shape2, pooling_size=[1, 2, 2, 1]) 
	with tf.variable_scope("full_connection") as full_layer3:
         layer2_output_flatten = tf.contrib.layers.flatten(layer2_output)
         layer3_output = tf.nn.relu(linear_layer(data=layer2_output_flatten,weights_size=full_conn_w_shape, biases_size=full_conn_b_shape))
        with tf.variable_scope("output") as output_layer4:
         output = linear_layer(data=layer3_output,weights_size=out_w_shape,biases_size=out_b_shape)
	return output;
	
def train():
	batch_size = 40
	dataset = load_data('1.gif')
	train_set_x, train_set_y = dataset[0]
	vaild_Set_x, valid_set_y = dataset[1]
	test_set_x,test_set_y = dataset[2]

	X = tf.placeholder(tf.float32, [batch_size, 57 * 47])
    	Y = tf.placeholder(tf.float32, [batch_size, 40])

	predict = convolutional_neural_network(X)
	cost_fun = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict,labels=Y))
	optimizer = tf.train.AdamOptimizer(1e-2).minimize(cost_fun)
	
	with tf.Session() as session:		
 	  session.run(tf.global_variables_initializer())
	  best_loss = float('inf')
	  for epoch in range(10):
	    epoch_loss = 0
  	    for i in range(np.shape(train_set_x)[0] // batch_size):
		x = train_set_x[i * batch_size : (i+1) * batch_size]
		y = train_set_y[i * batch_size : (i+1) * batch_size]
		cost = session.run([optimizer,cost_fun],feed_dict={X:x,Y:y})
		#print cost
		cost =cost[1]
		epoch_loss += cost
	    print(epoch, ':',epoch_loss)
	  correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
	  test_pred = tf.argmax(predict,1).eval({X:test_set_x})
	  test_true = np.argmax(test_set_y,1)
	  test_correct = correct.eval({X: test_set_x, Y: test_set_y})
	  incorrect_index = [i for i in range(np.shape(test_correct)[0]) if not test_correct[i]]
	  for i in incorrect_index:
		print (' Right is %i ,but miss predict %i'%(test_true[i],test_pred[i]))

if __name__ == '__main__':
	train()



		
	
	
















	
		






																				























