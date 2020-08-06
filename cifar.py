import collections
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import argparse
import sys
from math import *


class Block(collections.namedtuple('Block',['scope','unit_fn','args'])):
	'A namede tuple describing a ResNet block.'

def subsample(inputs,factor,scope=None):
	if factor == 1:
		return inputs
	else:
		return slim.max_pool2d(inputs,[1,1],stride=factor,scope=scope)

def open_conv2d(inputs,num_outputs,kernel_size,stride,scope=None,p=None,partial=None,tran=None):

	num_inputs = inputs.get_shape().as_list()[3]
	weight = variable_with_weight_loss(shape=[3,3,num_inputs,num_outputs],scope=scope,initializer=tf.contrib.layers.variance_scaling_initializer())
	weight = tf.transpose(weight,perm=[2,3,0,1])

	weight = get_coef(weight,partial)
	kernel = z2_kernel(weight,num_inputs,num_outputs,p=p,partial=partial,tran=tran)
	outputs = tf.nn.conv2d(inputs,kernel,strides=[1,stride,stride,1],padding='SAME',name='conv_op')

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

def variable_with_weight_loss(shape,wl=0.001,scope=None,initializer=None):
	var = tf.get_variable('weights',shape,initializer=initializer)
	weight_loss = tf.multiply(tf.nn.l2_loss(var),wl,name='weight_loss')
	tf.add_to_collection('losses',weight_loss)

	return var

def stack_blocks_dense(net,blocks,outputs_collections=None,is_training=True,p=None,partial=None,tran=None):
	for block in blocks:
		with tf.variable_scope(block.scope,'block',[net]) as sc:
			for i, unit in enumerate(block.args):
				with tf.variable_scope('unit_%d' %(i+1), values=[net]):
					unit_depth,unit_stride = unit 
					net = block.unit_fn(net,
										depth=unit_depth,
										stride=unit_stride,
										is_training=is_training,
										p=p,
										partial=partial,
										tran=tran)
	return net

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
	
	if kernel_size == 1:
		weight = variable_with_weight_loss(shape=[1,1,num_inputs*p,num_outputs],scope=scope,initializer=tf.contrib.layers.variance_scaling_initializer())
		weight = tf.reshape(weight,[1,1,num_inputs,p,num_outputs])

		kernel_list = [tf.concat([weight[:,:,:,-k:,:],weight[:,:,:,:-k,:]],axis=3) for k in range(p)]
		kernel = tf.stack(kernel_list,axis=-1)
		kernel = tf.reshape(kernel,[1,1,num_inputs*p,num_outputs*p])

	else:
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


	if stride == 1:
		outputs = tf.nn.conv2d(inputs,kernel,strides=[1,stride,stride,1],padding='SAME',name='conv_op')
	else:
		pad_total = kernel_size - 1
		pad_beg = pad_total // 2
		pad_end = pad_total - pad_beg
		inputs = tf.pad(inputs,[[0,0],[pad_beg,pad_end],[pad_beg,pad_end],[0,0]])
		outputs = tf.nn.conv2d(inputs,kernel,strides=[1,stride,stride,1],padding='VALID',name='conv_op')

	return outputs

def bottleneck(inputs,depth,stride,outputs_collections=None,scope=None,is_training=True,p=None,partial=None,tran=None):

	with tf.variable_scope('bottleneck') as sc:
		depth_in = int(inputs.get_shape().as_list()[3]/p)
		preact = g_bn(inputs,p,is_training=is_training)

		with tf.variable_scope('shortcut'):

			if depth == depth_in:
				shortcut = inputs
			else:
				shortcut = g_conv2d(preact,depth,1,stride,'shortcut',p,partial,tran)

		with tf.variable_scope('conv1'):
			conv1 = g_conv2d(preact,depth,5,stride,'conv1',p,partial,tran)
		mid_ac = g_bn(conv1,p,is_training=is_training)
		with tf.variable_scope('conv2'):
			conv2 = g_conv2d(mid_ac,depth,5,1,'conv2',p,partial,tran)


	output = shortcut + conv2

	return output

def resnet(inputs,blocks,num_classes=None,is_training=None,reuse=None,scope=None,p=8,partial=None,tran=None):
	net = inputs
	with tf.variable_scope('first_conv') as sc:
		net = open_conv2d(net,11,5,stride=1,scope=sc,p=p,partial=partial,tran=tran)
	
	net = stack_blocks_dense(net,blocks,is_training=is_training,p=p,partial=partial,tran=tran)
	net = g_bn(net,p,is_training=is_training)

	net = tf.reduce_mean(net,[1,2],name='pool5',keep_dims=True)

	num_channel = net.get_shape().as_list()[3]
	with tf.variable_scope('fc') as sc:
		weights = variable_with_weight_loss(shape=[1,1,num_channel,num_classes],initializer=tf.contrib.layers.xavier_initializer_conv2d())

		biases = tf.get_variable('biases',[num_classes],initializer=tf.constant_initializer(0.1))
		net = tf.nn.conv2d(net,weights,[1,1,1,1],padding='SAME')
		net = tf.nn.bias_add(net,biases)
	return net 

def resnet_simple(num_classes,batch_size,
	height,width,channal,reuse=None,scope='resnet_simple',n=None,p=None,partial=None,tran=None):
	with tf.name_scope('input'):
		input_x = tf.placeholder(tf.float32,[None,height,width,channal],name='input_x')
		input_y = tf.placeholder(tf.int32,[None],name='input_y')
		is_training = tf.placeholder(tf.bool,name='is_training')
		learning_rate = tf.placeholder(tf.float32,name='learning_rate')

	labels = input_y

	epoch_step = tf.Variable(0,trainable=False,name='Epoch_Step')
	iteration_step = tf.Variable(0,trainable=False,name='Iteration_Step')
	epoch_increment = tf.assign(epoch_step,tf.add(epoch_step,tf.constant(1)))
	iteration_increment = tf.assign(iteration_step,tf.add(epoch_step,tf.constant(1)))


	blocks=[
	Block('block1',bottleneck,[(11,1)] * n),
	Block('block2',bottleneck,[(23,2)]+[(23,1)]*(n-1)),
	Block('block3',bottleneck,[(45,2)]+[(45,1)]*(n-1))]
	logits = resnet(input_x,blocks,num_classes,is_training=is_training,reuse=reuse,scope=scope,p=p,partial=partial,tran=tran) 
	logits = tf.squeeze(logits,[1,2],name='squeeze')
	with tf.name_scope('loss_function'):
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='cross_entropy_per_example')
		cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
		tf.add_to_collection('losses',cross_entropy_mean)
		total_loss = tf.add_n(tf.get_collection('losses'),name= 'total_loss')

	with tf.name_scope('train_step'):
		optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			train_op = optimizer.minimize(total_loss)

	with tf.name_scope('get_accuracy'):
		predictions = tf.argmax(logits,axis=1,name='predictions')
		correct_prediction = tf.equal(tf.cast(predictions,tf.int32),input_y)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

	return learning_rate,input_x,input_y,is_training,epoch_step,epoch_increment,iteration_step,iteration_increment,total_loss,train_op,accuracy

def do_eval(sess,evalX,evalY,total_loss,accuracy,batch_size=100):
	number_examples = len(evalX)
	eval_loss,eval_acc,eval_counter = .0,.0,.0
	for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples+1,batch_size)):
		feed_dict = {input_x:evalX[start:end],input_y:evalY[start:end],is_training:False}
		curr_eval_loss,curr_eval_acc = sess.run([total_loss,accuracy],feed_dict=feed_dict)
		eval_loss,eval_acc,eval_counter = eval_loss+curr_eval_loss,eval_acc+curr_eval_acc,eval_counter+1
	return eval_loss/float(eval_counter),eval_acc/float(eval_counter)

def unpickle(file):
	import pickle
	with open(file,'rb') as fo:
		dict = pickle.load(fo,encoding='bytes')
	return dict

def load_cifar10():

	height,width = 32,32
	channal = 3
	train_data = np.zeros(shape=(0,height*width*channal))
	train_labels = np.zeros(shape=0)
	for i in range(5):
		file_train = '../../cifar10/data_batch_' + str(i+1)
		train_dict = unpickle(file_train)
		train_data = np.vstack((train_data,train_dict[b'data']))
		train_labels = np.hstack((train_labels,np.array(train_dict[b'labels'])))
	print('end load train data.')
	n_train = train_data.shape[0]
	train_data = np.reshape(train_data,[n_train,3,height,width])
	train_data = np.transpose(train_data,(0,2,3,1))
	file_test = '../../cifar10/test_batch'
	test_dict = unpickle(file_test)
	test_data = test_dict[b'data']
	test_labels = test_dict[b'labels']
	n_test = test_data.shape[0]
	test_data =np.reshape(test_data,[n_test,3,height,width])
	test_data = np.transpose(test_data,(0,2,3,1))
	return height,width,channal,train_data,train_labels,test_data,test_labels

def load_cifar100():

	height,width = 32,32
	channal = 3
	train_data = np.zeros(shape=(0,height*width*channal))
	train_labels = np.zeros(shape=0)
	file_train = '../../cifar100/train'
	train_dict = unpickle(file_train)
	train_data = train_dict[b'data']
	train_labels = train_dict[b'fine_labels']
	train_labels = np.array(train_labels)
	print('end load train data.')
	n_train = train_data.shape[0]
	train_data = np.reshape(train_data,[n_train,3,height,width])
	train_data = np.transpose(train_data,(0,2,3,1))
	file_test = '../../cifar100/test'
	test_dict = unpickle(file_test)
	test_data = test_dict[b'data']
	test_labels = test_dict[b'fine_labels']
	n_test = test_data.shape[0]
	test_data =np.reshape(test_data,[n_test,3,height,width])
	test_data = np.transpose(test_data,(0,2,3,1))
	return height,width,channal,train_data,train_labels,test_data,test_labels

def pre_treatment(train_data,train_labels,test_data,test_labels,valid_portion):
	n_train = len(train_data)
	trainX = train_data
	trainY = train_labels
	testX = test_data
	testY = test_labels
	return trainX,trainY,testX,testY

def data_normalization(train_data_raw, test_data_raw, normalize_type):
	if normalize_type == 'divide-255':
		train_data = train_data_raw/255.0
		test_data = test_data_raw/255.0

		return train_data, test_data

	elif normalize_type=='divide-256':
		train_data=train_data_raw/256.0
		test_data=test_data_raw/256.0

		return train_data, test_data

	elif normalize_type=='by-channels':
		train_data=np.zeros(train_data_raw.shape)
		test_data=np.zeros(test_data_raw.shape)
		for channel in range(train_data_raw.shape[-1]):
			images=np.concatenate((train_data_raw, test_data_raw), axis=0)
			channel_mean=np.mean(images[:,:,:,channel])
			channel_std=np.std(images[:,:,:,channel])
			train_data[:,:,:,channel]=(train_data_raw[:,:,:,channel]-channel_mean)/channel_std
			test_data[:,:,:,channel]=(test_data_raw[:,:,:,channel]-channel_mean)/channel_std

		return train_data, test_data

	elif normalize_type=='None':

		return train_data_raw, test_data_raw

def myshuffle(trainX,trainY):
	n_train = len(trainX)
	index = list(range(n_train))
	random.shuffle(index)
	trainX = trainX[index]
	trainY = trainY[index]
	return trainX,trainY

def augment_image(image, pad):
    """Perform zero padding, randomly crop image to original size,
    maybe mirror horizontally"""
    init_shape = image.shape
    new_shape = [init_shape[0] + pad * 2,
                 init_shape[1] + pad * 2,
                 init_shape[2]]
    zeros_padded = np.zeros(new_shape)
    zeros_padded[pad:init_shape[0] + pad, pad:init_shape[1] + pad, :] = image
    init_x = np.random.randint(0, pad * 2)
    init_y = np.random.randint(0, pad * 2)
    cropped = zeros_padded[
        init_x: init_x + init_shape[0],
        init_y: init_y + init_shape[1],
        :]
    flip = random.getrandbits(1)
    if flip:
        cropped = cropped[:, ::-1, :]
    return cropped

def augment_all_images(initial_images, pad):
    new_images = np.zeros(initial_images.shape)
    for i in range(initial_images.shape[0]):
        new_images[i] = augment_image(initial_images[i], pad=4)
    return new_images

def count_params():
    total_params=0
    for variable in tf.trainable_variables():
        shape=variable.get_shape()
        params=1
        for dim in shape:
            params=params*dim.value
        total_params+=params
    print("Total training params:",total_params)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--aug',default=True,type=bool)
	parser.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100'])
	parser.add_argument('--n_value',default=7)
	parser.add_argument('--p_value',default=8,choices=[1,4,8,12])
	parser.add_argument('--normalize_type',default='by-channels',choices=['by-channels','divide-255','divide-256'])
	parser.add_argument('--gpu',default=0,choices=[0,1])
	args = parser.parse_args()

	gpu_id = args.gpu
	CUDA_VISIBLE_DEVICES = gpu_id
	dataset = args.dataset
	if_aug = args.aug

	n_value = args.n_value
	p = args.p_value
	normalize_type = args.normalize_type 
	name = os.path.basename(sys.argv[0]).split(".")[0]

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


	if dataset == 'cifar10':
		height,width,channal,train_data,train_labels,test_data,test_labels = load_cifar10()
		num_classes = 10
	elif dataset == 'cifar100':
		height,width,channal,train_data,train_labels,test_data,test_labels = load_cifar100()
		num_classes = 100

	ckpt_dir = name + '_checkpoint/'

	if os.path.exists(ckpt_dir) == False:
		os.mkdir(ckpt_dir)

	train_data, test_data = data_normalization(train_data,test_data,normalize_type)
	valid_portion = .1
	trainX,trainY,testX,testY = pre_treatment(train_data,train_labels,test_data,test_labels,valid_portion)
	print('training dataset: %d' %(len(trainX)))
	print('test dataset: %d' %(len(testX)))

	print('end process data.')

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		origin_learning_rate = .1
		num_epoches = 300
		validate_every = 1
		save_every = 1
		batch_size = 128
		best_acc = 0.0

		learning_rate,input_x,input_y,is_training,epoch_step,epoch_increment,iteration_step,iteration_increment,total_loss,train_op,accuracy \
		= resnet_simple(num_classes,batch_size,height,width,channal,n=n_value,p=p,partial=partial_dict,tran=tran_to_partial_coef)

		saver = tf.train.Saver()
		
		if os.path.exists(ckpt_dir+'checkpoint'):
			print('Restoring Variables from checkpoint')
			saver.restore(sess,tf.train.latest_checkpoint(ckpt_dir))
		else:
			print('Initializing Variables')
			sess.run(tf.global_variables_initializer())

		if os.path.exists(name + '_result'):
			record_train_loss = np.load(name + '_result/train_loss.npy')
			record_train_acc = np.load(name + '_result/train_acc.npy')
			record_test_loss = np.load(name + '_result/test_loss.npy')
			record_test_acc = np.load(name + '_result/test_acc.npy')

		else:
			os.mkdir(name+'_result')
			record_train_loss = np.zeros(num_epoches)
			record_train_acc = np.zeros(num_epoches)
			record_test_loss = np.zeros(num_epoches)
			record_test_acc = np.zeros(num_epoches)
			np.save(name + '_result/train_loss.npy',record_train_loss)
			np.save(name + '_result/train_acc.npy',record_train_acc)
			np.save(name + '_result/test_loss.npy',record_test_loss)
			np.save(name + '_result/test_acc.npy',record_test_acc)

		count_params()
		curr_epoch = sess.run(epoch_step)

		number_of_training_data = len(trainX)
		for epoch in range(curr_epoch,num_epoches):

			if epoch < 2:
				cur_learning_rate = .01
			if epoch < 150:
				cur_learning_rate = origin_learning_rate
			elif epoch < 225:
				cur_learning_rate = origin_learning_rate * .1
			else:
				cur_learning_rate = origin_learning_rate * .01

			loss, acc, counter = .0, .0, 0
			trainX, trainY = myshuffle(trainX,trainY)
			if if_aug:
				cur_trainX = augment_all_images(trainX,pad=4) 
			else:
				cur_trainX = trainX
			for start,end in zip(range(0,number_of_training_data,batch_size),range(batch_size,number_of_training_data+1,batch_size)):
				feed_dict = {input_x:cur_trainX[start:end],input_y:trainY[start:end],is_training:True,learning_rate:cur_learning_rate}
				curr_loss,curr_acc,_ = sess.run([total_loss,accuracy,train_op],feed_dict)
				sess.run(iteration_increment)
				loss,counter,acc = loss+curr_loss,counter+1,acc+curr_acc
				display_num = 50
				if counter % display_num == 0:
					print('Epoch %d\t Iteration %d\t Train Loss:%.3e\t Train Accuracy:%.5f' %(epoch,counter,loss/float(display_num),acc/float(display_num)))
					record_train_loss[epoch] = loss/float(display_num)
					record_train_acc[epoch] = acc/float(display_num)
					loss,acc =.0,.0

			sess.run(epoch_increment)
			
			if epoch % save_every == 0:
				save_path = ckpt_dir+'model.ckpt'
				saver.save(sess,save_path,global_step=epoch)
				test_loss,test_acc = do_eval(sess,testX,testY,total_loss,accuracy)

				record_test_loss[epoch] = test_loss
				record_test_acc[epoch] = test_acc
				
				np.save(name + '_result/train_loss.npy',record_train_loss)
				np.save(name + '_result/train_acc.npy',record_train_acc)
				np.save(name + '_result/test_loss.npy',record_test_loss)
				np.save(name + '_result/test_acc.npy',record_test_acc)

				print('Epoch %d\tTest Loss:%.3e\tTest Accuracy:%.5f' %(epoch,test_loss,test_acc))
				if test_acc > best_acc:
					best_acc = test_acc
				print('The best Accuracy is %.5f' %(best_acc))







	