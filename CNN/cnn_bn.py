import scipy.io as sio
import time
import numpy as np

alldata = sio.loadmat('BN_str_0_bins_Ys.mat')
cb = alldata['Bins_0']
Y = alldata['Ytr_0']
tr = 10
va = 4
te = 17
zlayers = 11
x_train = np.array([cb[i*31:i*31+tr] for i in range(int(len(cb)/31))])
x_train = np.reshape(x_train,[80*tr,43,21,zlayers])
x_valid = np.array([cb[i*31+tr:i*31+tr+va] for i in range(int(len(cb)/31))])
x_valid = np.reshape(x_valid,[80*va,43,21,zlayers])
x_test = np.array([cb[i*31+tr+va:i*31+tr+va+te] for i in range(int(len(cb)/31))])
x_test = np.reshape(x_test,[80*te,43,21,zlayers])

Y_train = np.array([Y[i*31:i*31+tr] for i in range(int(len(Y)/31))])
Y_train = np.reshape(Y_train,[80*tr,1])
Y_valid = np.array([Y[i*31+tr:i*31+tr+va] for i in range(int(len(Y)/31))]) 
Y_valid = np.reshape(Y_valid,[80*va,1])
Y_test = np.array([Y[i*31+tr+va:i*31+tr+va+te] for i in range(int(len(Y)/31))])
Y_test = np.reshape(Y_test,[80*te,1])
#
import tensorflow as tf
sess = tf.InteractiveSession()


def weight_variable(shape):
	#
    initial = tf.truncated_normal(shape, stddev=0.1)
    W = tf.Variable(initial)
    #
    return W

def bias_variable(shape):
	#
    initial = tf.constant(0.1, shape=shape)
    b = tf.Variable(initial)
    #
    return b

def conv2d(x, W):
	#
    h_conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    #
    return h_conv

def max_pool_2x2(x):
	#
    h_max = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #
    return h_max


max_step = 25 

start_time = time.time() 

x = tf.placeholder(tf.float32, shape=[None,43,21,zlayers])
y_ = tf.placeholder(tf.float32, shape=[None,1])

W_conv1 = weight_variable([2, 2, zlayers, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([2, 2, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_pool2_flat = tf.reshape(h_pool2, [ -1,11*6*64])

W_fc1 = weight_variable([11*6*64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([1024, 512])
b_fc2 = bias_variable([512])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

W_fc3 = weight_variable([512, 256])
b_fc3 = bias_variable([256])
h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

keep_prob = tf.placeholder(tf.float32)
h_fc2_drop = tf.nn.dropout(h_fc3, keep_prob)

W_fc4 = weight_variable([256, 1])
b_fc4 = bias_variable([1])
y_conv = tf.matmul(h_fc2_drop, W_fc4) + b_fc4

cost = tf.reduce_mean(tf.square(y_conv - y_))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cost)

sess.run(tf.global_variables_initializer())
Wc1i = sess.run(W_conv1)
Wc2i = sess.run(W_conv2)
Wf1i = sess.run(W_fc1)
Wf2i = sess.run(W_fc2)
Wf3i = sess.run(W_fc3)
Wf4i = sess.run(W_fc4)
bc1i = sess.run(b_conv1)
bc2i = sess.run(b_conv2)
bf1i = sess.run(b_fc1)
bf2i = sess.run(b_fc2)
bf3i = sess.run(b_fc3)
bf4i = sess.run(b_fc4)
ppp = 0
valid_cost1 = 10000
trc = []
vac = []
tec = []
    # run the training
for i in range(max_step):
    #
    if i%1 == 0:
            # output the training accuracy every 100 iterations
        train_cost = cost.eval(session=sess,feed_dict={
            x:x_train, y_:Y_train, keep_prob: 1.0})
        print("step %d, training cost %g"%(i, train_cost))
        trc.append(train_cost)
        valid_cost = cost.eval(session=sess,feed_dict={
            x:x_valid, y_:Y_valid, keep_prob: 1.0})
        print("step %d, validation cost %g"%(i, valid_cost))
        vac.append(valid_cost)
        test_cost = cost.eval(session=sess,feed_dict={
            x:x_test, y_:Y_test, keep_prob: 1.0})
        print("step %d, test cost %g"%(i, test_cost))
        tec.append(test_cost)
        if valid_cost < valid_cost1:
            valid_cost1 = valid_cost
            ppp = ppp + 1
            print("Yayyy point: " + str(ppp))
            Wc1 = sess.run(W_conv1)
            Wc2 = sess.run(W_conv2)
            Wf1 = sess.run(W_fc1)
            Wf2 = sess.run(W_fc2)
            Wf3 = sess.run(W_fc3)
            Wf4 = sess.run(W_fc4)
            bc1 = sess.run(b_conv1)
            bc2 = sess.run(b_conv2)
            bf1 = sess.run(b_fc1)
            bf2 = sess.run(b_fc2)
            bf3 = sess.run(b_fc3)
            bf4 = sess.run(b_fc4)                                                     
    train_step.run(feed_dict={x: x_train, y_: Y_train, keep_prob: 0.5})
    #
            # Update the events file which is used to monitor the training (in this case,
    #

stop_time = time.time()
print('The training takes %f second to finish'%(stop_time - start_time))
sess.run(W_conv1.assign(Wc1))
sess.run(W_conv2.assign(Wc2))
sess.run(b_conv1.assign(bc1))
sess.run(b_conv2.assign(bc2))
sess.run(W_fc1.assign(Wf1))
sess.run(W_fc2.assign(Wf2))
sess.run(W_fc3.assign(Wf3))
sess.run(W_fc4.assign(Wf4))
sess.run(b_fc1.assign(bf1))
sess.run(b_fc2.assign(bf2))
sess.run(b_fc3.assign(bf3))
sess.run(b_fc4.assign(bf4))
test_cost = cost.eval(feed_dict={
            x:x_test, y_:Y_test, keep_prob: 1.0})
print("step %d, test cost %g"%(i, test_cost))
sio.savemat('training_cost.mat',{'tr_c': trc})
#np.savetxt('training_cost.txt',trc,fmt='%s',delimiter='\t')
sio.savemat('validation_cost.mat',{'va_c': vac})#
#np.savetxt('validation_cost.txt',vac,fmt='%s',delimiter='\t')
sio.savemat('test_cost.mat',{'te_c': tec})
#np.savetxt('test_cost.txt',tec,fmt='%s',delimiter='\t')
true_ans1 = sess.run(y_,{x: x_test, y_: Y_test, keep_prob: 1})
pred_ans1 = sess.run(y_conv,{x: x_test, y_: Y_test, keep_prob: 1})
print(np.mean(np.abs(true_ans1-pred_ans1))) 
true_mean = np.mean(true_ans1)
M = true_ans1 - true_mean*np.ones((true_ans1.size,1))
E = true_ans1 - pred_ans1
R2 = 1 - (np.sum(E*E))/(np.sum(M*M))
print(R2)
summary = np.array([(true_ans1[i][0],pred_ans1[i][0]) for i in range(len(true_ans1))])
sio.savemat('test_accuracy.mat',{'Rte': summary})
#np.savetxt('results_te.txt',summary,fmt='%s',delimiter='\t')
true_ans1 = sess.run(y_,{x: x_valid, y_: Y_valid, keep_prob: 1})
pred_ans1 = sess.run(y_conv,{x: x_valid, y_: Y_valid, keep_prob: 1})
print(np.mean(np.abs(true_ans1-pred_ans1))) 
true_mean = np.mean(true_ans1)
M = true_ans1 - true_mean*np.ones((true_ans1.size,1))
E = true_ans1 - pred_ans1
R2 = 1 - (np.sum(E*E))/(np.sum(M*M))
print(R2)
summary = np.array([(true_ans1[i][0],pred_ans1[i][0]) for i in range(len(true_ans1))])
sio.savemat('validation_accuracy.mat',{'Rva': summary})
#np.savetxt('results_va.txt',summary,fmt='%s',delimiter='\t')
true_ans1 = sess.run(y_,{x: x_train, y_: Y_train, keep_prob: 1})
pred_ans1 = sess.run(y_conv,{x: x_train, y_: Y_train, keep_prob: 1})
print(np.mean(np.abs(true_ans1-pred_ans1))) 
true_mean = np.mean(true_ans1)
M = true_ans1 - true_mean*np.ones((true_ans1.size,1))
E = true_ans1 - pred_ans1
R2 = 1 - (np.sum(E*E))/(np.sum(M*M))
print(R2)
summary = np.array([(true_ans1[i][0],pred_ans1[i][0]) for i in range(len(true_ans1))])
sio.savemat('training_accuracy.mat',{'Rtr': summary})
#np.savetxt('results_tr.txt',summary,fmt='%s',delimiter='\t')
sio.savemat('wc1.mat',{'Wc1': Wc1})
sio.savemat('wc2.mat',{'Wc2': Wc2})
sio.savemat('wf1.mat',{'Wf1': Wf1})
sio.savemat('wf2.mat',{'Wf2': Wf2})
sio.savemat('wf3.mat',{'Wf3': Wf3})
sio.savemat('wf4.mat',{'Wf4': Wf4})
sio.savemat('bc1.mat',{'bc1': bc1})
sio.savemat('bc2.mat',{'bc2': bc2})
sio.savemat('bf1.mat',{'bf1': bf1})
sio.savemat('bf2.mat',{'bf2': bf2})
sio.savemat('bf3.mat',{'bf3': bf3})
sio.savemat('bf4.mat',{'bf4': bf4})
sio.savemat('wc1i.mat',{'Wc1i': Wc1i})
sio.savemat('wc2i.mat',{'Wc2i': Wc2i})
sio.savemat('wf1i.mat',{'Wf1i': Wf1i})
sio.savemat('wf2i.mat',{'Wf2i': Wf2i})
sio.savemat('wf3i.mat',{'Wf3i': Wf3i})
sio.savemat('wf4i.mat',{'Wf4i': Wf4i})
sio.savemat('bc1i.mat',{'bc1i': bc1i})
sio.savemat('bc2i.mat',{'bc2i': bc2i})
sio.savemat('bf1i.mat',{'bf1i': bf1i})
sio.savemat('bf2i.mat',{'bf2i': bf2i})
sio.savemat('bf3i.mat',{'bf3i': bf3i})
sio.savemat('bf4i.mat',{'bf4i': bf4i})