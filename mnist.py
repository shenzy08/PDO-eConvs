import collections
import tensorflow as tf
import numpy as np
import os
import random
import argparse
import sys
from math import *

def open_conv2d(inputs,num_outputs,kernel_size,stride,scope=None,p=None,partial=None,tran=None):

	num_inputs = inputs.get_shape().as_list()[3]
	weight = variable_with_weight_loss(shape=[3,3,num_inputs,num_outputs],scope=scope,initializer=tf.contrib.layers.variance_scaling_initializer())
	weight = tf.transpose(weight,perm=[2,3,0,1])

	weight = get_coef(weight,partial)
	kernel = z2_kernel(weight,num_inputs,num_outputs,p=p,partial=partial,tran=tran)

	inputs = tf.pad(inputs,[[0,0],[1,1],[1,1],[0,0]])

	outputs = tf.nn.conv2d(inputs,kernel,strides=[1,stride,stride,1],padding='VALID',name='conv_op')


	return outputs

def get_coef(weight,partial):

	num_inputs,num_outputs = weight.get_shape().as_list()[:2]

	partial = np.array(partial)
	transformation = partial[[0,1,2,3,4,5,7,8,12],1:4,1:4]
	transformation = np.reshape(transformation,[9,9])

	weight = tf.reshape(weight,[-1,9])

	inv_transformation = tf.constant(np.linalg.inv(transformation),dtype=tf.float32)
	weight = tf.matmul(weight,inv_transformation)
	weight = tf.reshape(weight,[num_inputs,num_outputs,9])

	return weight

def z2_kernel(weight,num_inputs,num_outputs,p,partial,tran):

	og_coef = tf.reshape(weight,[num_inputs*num_outputs,9])
	tran_to_partial_coef = [tf.constant(a,dtype=tf.float32) for a in tran]


	partial_coef = [tf.matmul(og_coef,a) for a in tran_to_partial_coef] 
	
	partial_dict = tf.constant(partial)
	partial_dict = tf.reshape(partial_dict,[15,25])

	kernel = [tf.matmul(a,partial_dict) for a in partial_coef] 
	kernel = tf.stack(kernel,axis=1)
	kernel = tf.reshape(kernel,[num_inputs,num_outputs*p,5,5])
	kernel = tf.transpose(kernel,perm=[2,3,0,1])
	
	return kernel

def variable_with_weight_loss(shape,wl=1e-2,scope=None,initializer=None):
	var = tf.get_variable('weights',shape,initializer=initializer)
	weight_loss = tf.multiply(tf.nn.l2_loss(var),wl,name='weight_loss')
	tf.add_to_collection('losses',weight_loss)

	return var

def g_bn(inputs,p,is_training):

	height,width,channel = inputs.get_shape().as_list()[1:]
	inputs = tf.reshape(inputs,[-1,height,width,int(channel/p),p])
	inputs = tf.transpose(inputs,perm=[0,1,2,4,3])
	inputs = tf.reshape(inputs,[-1,height,width*p,int(channel/p)])
	outputs = tf.contrib.layers.batch_norm(inputs,scale=True,activation_fn=tf.nn.relu,is_training=is_training)
	outputs = tf.reshape(outputs,[-1,height,width,p,int(channel/p)])
	outputs = tf.transpose(outputs,[0,1,2,4,3])
	outputs = tf.reshape(outputs,[-1,height,width,channel])
	return outputs

def g_conv2d(inputs,num_outputs,kernel_size,stride,scope=None,p=None,partial=None,tran=None):
	
	num_inputs = int(inputs.get_shape().as_list()[3]/p)

	weight = variable_with_weight_loss(shape=[3,3,num_inputs*p,num_outputs],scope=scope,initializer=tf.contrib.layers.variance_scaling_initializer())
	weight = tf.transpose(weight,perm=[2,3,0,1])
	weight = get_coef(weight,partial)


	og_coef = tf.reshape(weight,[num_inputs*p*num_outputs,9])
	tran_to_partial_coef = [tf.constant(a,dtype=tf.float32) for a in tran]
	partial_coef = [tf.matmul(og_coef,a) for a in tran_to_partial_coef] 

	partial_dict = tf.constant(partial)
	partial_dict = tf.reshape(partial_dict,[15,25])

	og_kernel_list = [tf.matmul(a,partial_dict) for a in partial_coef] 
	og_kernel_list = [tf.reshape(og_kernel,[num_inputs,p,num_outputs,25]) for og_kernel in og_kernel_list] 
	og_kernel_list = [tf.concat([og_kernel_list[k][:,-k:,:],og_kernel_list[k][:,:-k,:]],axis=1) for k in range(p)]

	kernel = tf.stack(og_kernel_list,axis=3)
	kernel = tf.reshape(kernel,[num_inputs*p,num_outputs*p,5,5])
	kernel = tf.transpose(kernel,perm=[2,3,0,1])

	inputs = tf.pad(inputs,[[0,0],[1,1],[1,1],[0,0]])

	outputs = tf.nn.conv2d(inputs,kernel,strides=[1,stride,stride,1],padding='VALID',name='conv_op')

	return outputs


def load_rot_mnist():
	f = open('../mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat')
	train_str = f.read()
	train_list = train_str.split()
	num_train = len(train_list)
	train = [float(x) for x in train_list]
	trainX = [train[i] for i in range(num_train) if (i+1)%785 != 0]
	trainX = np.reshape(trainX,[12000,28,28,1])
	trainY = [int(train[i]) for i in range(num_train) if (i+1)%785 == 0]
	trainY = np.array(trainY)
	f.close()

	f = open('../mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat')
	test_str = f.read()
	test_list = test_str.split()
	num_test = len(test_list)
	test = [float(x) for x in test_list]
	testX = [test[i] for i in range(num_test) if (i+1)%785 != 0]
	testX = np.reshape(testX,[50000,28,28,1])
	testY = [int(test[i]) for i in range(num_test) if (i+1)%785 == 0]
	testY = np.array(testY)
	f.close()

	return trainX, trainY, testX, testY

def CNN_simple(num_classes,batch_size,height,width,channal,p,partial,tran):
	with tf.name_scope('input'):
		input_x = tf.placeholder(tf.float32,[None,height,width,channal],name='input_x')
		input_y = tf.placeholder(tf.int32,[None],name='input_y')
		is_training = tf.placeholder(tf.bool,name='is_training')
		learning_rate = tf.placeholder(tf.float32,name='learning_rate')
		keep_prob = tf.placeholder(tf.float32,name='keep_prob')

	labels = input_y

	epoch_step = tf.Variable(0,trainable=False,name='Epoch_Step')
	iteration_step = tf.Variable(0,trainable=False,name='Iteration_Step')
	epoch_increment = tf.assign(epoch_step,tf.add(epoch_step,tf.constant(1)))
	iteration_increment = tf.assign(iteration_step,tf.add(epoch_step,tf.constant(1)))
	
	with tf.variable_scope('open_conv') as sc:

		current = open_conv2d(input_x,7,5,1,scope=None,p=p,partial=partial,tran=tran)
		current = g_bn(current,p,is_training)
		current = tf.nn.dropout(current,keep_prob)

	with tf.variable_scope('second_conv') as sc:
		current = g_conv2d(current,7,5,1,None,p,partial,tran)
		current = g_bn(current,p,is_training)
		current = tf.nn.max_pool(current,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

	for i in range(4):
		with tf.variable_scope('conv'+str(i+3)) as sc:
			current = g_conv2d(current,7,5,1,None,p,partial,tran)
			current = g_bn(current,p,is_training)
			current = tf.nn.dropout(current,keep_prob)

	with tf.variable_scope('fc') as sc:
		weight = variable_with_weight_loss(shape=[4,4,7*8,10],scope=None,initializer=tf.contrib.layers.xavier_initializer_conv2d())
		current = tf.nn.conv2d(current,weight,strides=[1,1,1,1],padding='VALID')
	logits = tf.squeeze(current,[1,2],name='squeeze')

	with tf.name_scope('loss_function'):
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='cross_entropy_per_example')
		cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
		tf.add_to_collection('losses',cross_entropy_mean)
		total_loss = tf.add_n(tf.get_collection('losses'),name= 'total_loss')

	with tf.name_scope('train_step'):
		optimizer = tf.train.AdamOptimizer(learning_rate)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			train_op = optimizer.minimize(total_loss)

	with tf.name_scope('get_accuracy'):
		predictions = tf.argmax(logits,axis=1,name='predictions')
		correct_prediction = tf.equal(tf.cast(predictions,tf.int32),input_y)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

	return learning_rate,input_x,input_y,is_training,keep_prob,epoch_step,epoch_increment,iteration_step,iteration_increment,total_loss,train_op,accuracy


def data_normalization(train_data_raw, test_data_raw):

	train_data=np.zeros(train_data_raw.shape)
	test_data=np.zeros(test_data_raw.shape)
	for channel in range(train_data_raw.shape[-1]):
		images = train_data_raw
		channel_mean=np.mean(images[:,:,:,channel])
		channel_std=np.std(images[:,:,:,channel])
		train_data[:,:,:,channel]=(train_data_raw[:,:,:,channel]-channel_mean)/channel_std
		test_data[:,:,:,channel]=(test_data_raw[:,:,:,channel]-channel_mean)/channel_std

	return train_data, test_data

def myshuffle(trainX,trainY):
	n_train = len(trainX)
	index = list(range(n_train))
	random.shuffle(index)
	trainX = trainX[index]
	trainY = trainY[index]
	return trainX,trainY

def count_params():
    total_params=0
    for variable in tf.trainable_variables():
        shape=variable.get_shape()
        params=1
        for dim in shape:
            params=params*dim.value
        total_params+=params
    print("Total training params:",total_params)

def do_eval(sess,evalX,evalY,total_loss,accuracy,batch_size=100):
	number_examples = len(evalX)
	eval_loss,eval_acc,eval_counter = .0,.0,.0
	for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples+1,batch_size)):
		feed_dict = {input_x:evalX[start:end],input_y:evalY[start:end],is_training:False,keep_prob:1.0}
		curr_eval_loss,curr_eval_acc = sess.run([total_loss,accuracy],feed_dict=feed_dict)
		eval_loss,eval_acc,eval_counter = eval_loss+curr_eval_loss,eval_acc+curr_eval_acc,eval_counter+1
	return eval_loss/float(eval_counter),eval_acc/float(eval_counter)

if __name__ == '__main__':

	origin_learning_rate = .001
	num_epoches = 200
	batch_size = 128
	my_keep_prob = .8
	best_acc = .0

	num_classes = 10
	height = 28
	width = 28
	channal = 1
	p = 8

	trainX, trainY, testX, testY = load_rot_mnist()
	trainX, testX = data_normalization(trainX,testX)

	name = os.path.basename(sys.argv[0]).split(".")[0]
	ckpt_dir = name + '_checkpoint/'
	partial_dict = [[[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]],
					[[0,0,0,0,0],[0,0,0,0,0],[0,-1/2,0,1/2,0],[0,0,0,0,0],[0,0,0,0,0]],
					[[0,0,0,0,0],[0,0,1/2,0,0],[0,0,0,0,0],[0,0,-1/2,0,0],[0,0,0,0,0]],
					[[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]],
					[[0,0,0,0,0],[0,-1/4,0,1/4,0],[0,0,0,0,0],[0,1/4,0,-1/4,0],[0,0,0,0,0]],
					[[0,0,0,0,0],[0,0,1,0,0],[0,0,-2,0,0],[0,0,1,0,0],[0,0,0,0,0]],
					[[0,0,0,0,0],[0,0,0,0,0],[-1/2,1,0,-1,1/2],[0,0,0,0,0],[0,0,0,0,0]],
					[[0,0,0,0,0],[0,1/2,-1,1/2,0],[0,0,0,0,0],[0,-1/2,1,-1/2,0],[0,0,0,0,0]],
					[[0,0,0,0,0],[0,-1/2,0,1/2,0],[0,1,0,-1,0],[0,-1/2,0,1/2,0],[0,0,0,0,0]],
					[[0,0,1/2,0,0],[0,0,-1,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,-1/2,0,0]],
					[[0,0,0,0,0],[0,0,0,0,0],[1,-4,6,-4,1],[0,0,0,0,0],[0,0,0,0,0]],
					[[0,0,0,0,0],[-1/4,1/2,0,-1/2,1/4],[0,0,0,0,0],[1/4,-1/2,0,1/2,-1/4],[0,0,0,0,0]],
					[[0,0,0,0,0],[0,1,-2,1,0],[0,-2,4,-2,0],[0,1,-2,1,0],[0,0,0,0,0]],
					[[0,-1/4,0,1/4,0],[0,1/2,0,-1/2,0],[0,0,0,0,0],[0,-1/2,0,1/2,0],[0,1/4,0,-1/4,0]],
					[[0,0,1,0,0],[0,0,-4,0,0],[0,0,6,0,0],[0,0,-4,0,0],[0,0,1,0,0]]]


	group_angle = [2*k*pi/p+pi/8 for k in range(p)]
	tran_to_partial_coef = [np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
									 [0,cos(x),sin(x),0,0,0,0,0,0,0,0,0,0,0,0],
									 [0,-sin(x),cos(x),0,0,0,0,0,0,0,0,0,0,0,0],
									 [0,0,0,pow(cos(x),2),2*cos(x)*sin(x),pow(sin(x),2),0,0,0,0,0,0,0,0,0],
									 [0,0,0,-cos(x)*sin(x),pow(cos(x),2)-pow(sin(x),2),sin(x)*cos(x),0,0,0,0,0,0,0,0,0],
									 [0,0,0,pow(sin(x),2),-2*cos(x)*sin(x),pow(cos(x),2),0,0,0,0,0,0,0,0,0],
									 [0,0,0,0,0,0,-pow(cos(x),2)*sin(x),pow(cos(x),3)-2*cos(x)*pow(sin(x),2),-pow(sin(x),3)+2*pow(cos(x),2)*sin(x), pow(sin(x),2)*cos(x),0,0,0,0,0],
									 [0,0,0,0,0,0,cos(x)*pow(sin(x),2),-2*pow(cos(x),2)*sin(x)+pow(sin(x),3),pow(cos(x),3)-2*cos(x)*pow(sin(x),2),sin(x)*pow(cos(x),2),0,0,0,0,0],
									 [0,0,0,0,0,0,0,0,0,0,pow(sin(x),2)*pow(cos(x),2),-2*pow(cos(x),3)*sin(x)+2*cos(x)*pow(sin(x),3),pow(cos(x),4)-4*pow(cos(x),2)*pow(sin(x),2)+pow(sin(x),4),-2*cos(x)*pow(sin(x),3)+2*pow(cos(x),3)*sin(x),pow(sin(x),2)*pow(cos(x),2)]]) for x in group_angle]

	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:

		learning_rate,input_x,input_y,is_training,keep_prob,epoch_step,epoch_increment,iteration_step,iteration_increment,total_loss,train_op,accuracy \
		= CNN_simple(num_classes,batch_size,height,width,channal,p=p,partial=partial_dict,tran=tran_to_partial_coef)

		sess.run(tf.global_variables_initializer())

		count_params()
		curr_epoch = sess.run(epoch_step)

		number_of_training_data = len(trainX)
		print('training samples:',number_of_training_data)
		print('test samples:',len(testX))
		for epoch in range(curr_epoch,num_epoches):

			if epoch <= 5:
				cur_learning_rate = .0001
			elif epoch < 100:
				cur_learning_rate = origin_learning_rate
			elif epoch < 150:
				cur_learning_rate = origin_learning_rate * .1
			else:
				cur_learning_rate = origin_learning_rate * .01

			loss, acc, counter = .0, .0, 0
			cur_trainX, cur_trainY = myshuffle(trainX,trainY)

			for start,end in zip(range(0,number_of_training_data,batch_size),range(batch_size,number_of_training_data+1,batch_size)):
				feed_dict = {input_x:cur_trainX[start:end],input_y:cur_trainY[start:end],is_training:True,learning_rate:cur_learning_rate,keep_prob:my_keep_prob}
				curr_loss,curr_acc,_ = sess.run([total_loss,accuracy,train_op],feed_dict)
				sess.run(iteration_increment)
				loss,counter,acc = loss+curr_loss,counter+1,acc+curr_acc
				display_num = 35
				if counter % display_num == 0:
					print('Epoch %d\t Iteration %d\t Train Loss:%.3e\t Train Accuracy:%.5f' %(epoch,counter,loss/float(display_num),acc/float(display_num)))
					loss,acc =.0,.0
			train_loss,train_acc = do_eval(sess,cur_trainX,cur_trainY,total_loss,accuracy)
			print('Epoch %d\tTrain Loss:%.3e\tTrain Accuracy:%.5f' %(epoch,train_loss,train_acc))


			sess.run(epoch_increment)

			test_loss,test_acc = do_eval(sess,testX,testY,total_loss,accuracy)

			print('Epoch %d\tTest Loss:%.3e\tTest Accuracy:%.5f' %(epoch,test_loss,test_acc))
			if test_acc > best_acc:
				best_acc = test_acc
			print('The best Accuracy is %.5f' %(best_acc))
