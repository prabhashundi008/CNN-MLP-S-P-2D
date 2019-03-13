#%% Neural Networks
import scipy.io as sio
import time
import numpy as np
import tensorflow as tf

alldata = sio.loadmat('BN_str_0_holes_Ys.mat')
T1 = alldata['St3c_0']
T2 = alldata['saA_0']
T = np.concatenate([T1,T2],axis=1)
feature_dim = T.shape[1]
Y = alldata['Ytr_0']
tr = 12
va = 4
te = 15
zlayers = 11
x_train = np.array([T[i*31:i*31+tr] for i in range(int(len(T)/31))])
x_train = np.reshape(x_train,[80*tr,feature_dim])
x_valid = np.array([T[i*31+tr:i*31+tr+va] for i in range(int(len(T)/31))])
x_valid = np.reshape(x_valid,[80*va,feature_dim])
x_test = np.array([T[i*31+tr+va:i*31+tr+va+te] for i in range(int(len(T)/31))])
x_test = np.reshape(x_test,[80*te,feature_dim])

Y_train = np.array([Y[i*31:i*31+tr] for i in range(int(len(Y)/31))])
Y_train = np.reshape(Y_train,[80*tr,1])
Y_valid = np.array([Y[i*31+tr:i*31+tr+va] for i in range(int(len(Y)/31))]) 
Y_valid = np.reshape(Y_valid,[80*va,1])
Y_test = np.array([Y[i*31+tr+va:i*31+tr+va+te] for i in range(int(len(Y)/31))])
Y_test = np.reshape(Y_test,[80*te,1])

def new_weights(shape):
#    return tf.Variable(tf.constant(0.0,shape = shape))
    return tf.Variable(tf.truncated_normal(shape,stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05,shape = [length]))

n1 = feature_dim; n2 = 25; n3 = 20; n4 = 15; n5 = 8; n6 = 3;# n7 = 8;# n8 = 3;

x = tf.placeholder(dtype=tf.float32,shape=(None,n1))
y1_ = tf.placeholder(dtype=tf.float32,shape=(None,1))

W1 = new_weights((n1,n2))
B1 = new_biases(n2)

W2 = new_weights((n2,n3))
B2 = new_biases(n3)
#
W3 = new_weights((n3,n4))
B3 = new_biases(n4)
###
W4 = new_weights((n4,n5))
B4 = new_biases(n5)
#
W5 = new_weights((n5,n6))
B5 = new_biases(n6)

W6 = new_weights((n6,1))
B6 = new_biases(1)

#W7 = new_weights((n7,n8))
#B7 = new_biases(n8)

#W8 = new_weights((n8,1))
#B8 = new_biases(1)

layer1 = tf.matmul(x,W1)+B1
layer1 = tf.nn.relu(layer1)
layer2 = tf.matmul(layer1,W2)+B2
layer2 = tf.nn.relu(layer2)
layer3 = tf.matmul(layer2,W3)+B3
layer3 = tf.nn.relu(layer3)
layer4 = tf.matmul(layer3,W4)+B4
layer4 = tf.nn.relu(layer4)
layer5 = tf.matmul(layer4,W5)+B5
layer5 = tf.nn.relu(layer5)
y1 = tf.matmul(layer5,W6)+B6
#layer6 = tf.nn.relu(layer6)
#layer7 = tf.matmul(layer6,W7)+B7
#layer7 = tf.nn.relu(layer7)
#y1 = tf.matmul(layer7,W8)+B8
#y1 = tf.nn.relu(y1)
#y2 = tf.matmul(layer5,W7)+B7
#y2 = tf.nn.relu(y2)
trc = []
vac = []
tec = []
cost1 = tf.square(y1-y1_)
cost = tf.reduce_mean(cost1)
train_step = tf.train.AdamOptimizer(1e-2).minimize(cost)
sess = tf.InteractiveSession()
valid_cost_bigloop = 10000
for ii in range(20):
    sess.run(tf.global_variables_initializer())

    Wf1i = sess.run(W1)
    Wf2i = sess.run(W2)
    Wf3i = sess.run(W3)
    Wf4i = sess.run(W4)
    Wf5i = sess.run(W5)
    Wf6i = sess.run(W6)
    bf1i = sess.run(B1)
    bf2i = sess.run(B2)
    bf3i = sess.run(B3)
    bf4i = sess.run(B4)
    bf5i = sess.run(B5)
    bf6i = sess.run(B6)

    valid_cost = 10000

    for i in range(25000):
        sess.run(train_step,{x:x_train,y1_:Y_train})
        if i%50 == 0:
            train_cost1 = sess.run(cost,{x:x_train,y1_:Y_train})
            trc.append(train_cost1)
           # print('Steps ',i,' Train Cost ',train_cost1)
            valid_cost1 = sess.run(cost,{x:x_valid,y1_:Y_valid})
            vac.append(valid_cost1)
           # print('Steps ',i,' Valid Cost ',valid_cost1)
            test_cost1 = sess.run(cost,{x:x_test,y1_:Y_test})
            tec.append(test_cost1)
           # print('Steps ',i,' Test Cost ',test_cost1)
            if valid_cost1 > valid_cost:
                break
            valid_cost = valid_cost1
        
    print(' Iter: ',ii,' Valid Cost: ',valid_cost1,' Test Cost: ',test_cost1)       
    if valid_cost < valid_cost_bigloop:  
        Wf1 = sess.run(W1)
        Wf2 = sess.run(W2)
        Wf3 = sess.run(W3)
        Wf4 = sess.run(W4)
        Wf5 = sess.run(W5)
        Wf6 = sess.run(W6)
        bf1 = sess.run(B1)
        bf2 = sess.run(B2)
        bf3 = sess.run(B3)
        bf4 = sess.run(B4)
        bf5 = sess.run(B5)
        bf6 = sess.run(B6)
        valid_cost_bigloop = valid_cost

sess.run(W1.assign(Wf1))
sess.run(W2.assign(Wf2))
sess.run(W3.assign(Wf3))
sess.run(W4.assign(Wf4))
sess.run(W5.assign(Wf5))
sess.run(W6.assign(Wf6))
sess.run(B1.assign(bf1))
sess.run(B2.assign(bf2))
sess.run(B3.assign(bf3))
sess.run(B4.assign(bf4))
sess.run(B5.assign(bf5))
sess.run(B6.assign(bf6))

test_cost1 = sess.run(cost,{x:x_test,y1_:Y_test})
print(' Final Test Cost: ', test_cost1 )

c_a_dict = {}
c_a_dict['tr_c'] = trc
c_a_dict['va_c'] = vac
c_a_dict['te_c'] = tec
weight_dict = {}
weight_dict['Wf1'] = Wf1
weight_dict['Wf2'] = Wf2
weight_dict['Wf3'] = Wf3
weight_dict['Wf4'] = Wf4
weight_dict['Wf5'] = Wf5
weight_dict['Wf6'] = Wf6
weight_dict['bf1'] = bf1
weight_dict['bf2'] = bf2
weight_dict['bf3'] = bf3
weight_dict['bf4'] = bf4
weight_dict['bf5'] = bf5
weight_dict['bf6'] = bf6
weight_dict['Wf1i'] = Wf1i
weight_dict['Wf2i'] = Wf2i
weight_dict['Wf3i'] = Wf3i
weight_dict['Wf4i'] = Wf4i
weight_dict['Wf5i'] = Wf5i
weight_dict['Wf6i'] = Wf6i
weight_dict['bf1i'] = bf1i
weight_dict['bf2i'] = bf2i
weight_dict['bf3i'] = bf3i
weight_dict['bf4i'] = bf4i
weight_dict['bf5i'] = bf5i
weight_dict['bf6i'] = bf6i


true_ans1 = sess.run(y1_,{x: x_test, y1_: Y_test})
pred_ans1 = sess.run(y1,{x: x_test, y1_: Y_test})
print(np.mean(np.abs(true_ans1-pred_ans1))) 
true_mean = np.mean(true_ans1)
M = true_ans1 - true_mean*np.ones((true_ans1.size,1))
E = true_ans1 - pred_ans1
R2 = 1 - (np.sum(E*E))/(np.sum(M*M))
print(R2)
summary = np.array([(true_ans1[i][0],pred_ans1[i][0]) for i in range(len(true_ans1))])
true_te = np.reshape(true_ans1,[80,te])
pred_te = np.reshape(pred_ans1,[80,te])
c_a_dict['Rte'] = summary

true_ans1 = sess.run(y1_,{x: x_valid, y1_: Y_valid})
pred_ans1 = sess.run(y1,{x: x_valid, y1_: Y_valid})
print(np.mean(np.abs(true_ans1-pred_ans1))) 
true_mean = np.mean(true_ans1)
M = true_ans1 - true_mean*np.ones((true_ans1.size,1))
E = true_ans1 - pred_ans1
R2 = 1 - (np.sum(E*E))/(np.sum(M*M))
print(R2)
summary = np.array([(true_ans1[i][0],pred_ans1[i][0]) for i in range(len(true_ans1))])
true_va = np.reshape(true_ans1,[80,va])
pred_va = np.reshape(pred_ans1,[80,va])
c_a_dict['Rva'] = summary

true_ans1 = sess.run(y1_,{x: x_train, y1_: Y_train})
pred_ans1 = sess.run(y1,{x: x_train, y1_: Y_train})
print(np.mean(np.abs(true_ans1-pred_ans1))) 
true_mean = np.mean(true_ans1)
M = true_ans1 - true_mean*np.ones((true_ans1.size,1))
E = true_ans1 - pred_ans1
R2 = 1 - (np.sum(E*E))/(np.sum(M*M))
print(R2)
summary = np.array([(true_ans1[i][0],pred_ans1[i][0]) for i in range(len(true_ans1))])
true_tr = np.reshape(true_ans1,[80,tr])
pred_tr = np.reshape(pred_ans1,[80,tr])
c_a_dict['Rtr'] = summary

true_tot = np.concatenate([true_tr,true_va,true_te],axis=1)
pred_tot = np.concatenate([pred_tr,pred_va,pred_te],axis=1)
c_a_dict['R_true'] = true_tot.T
c_a_dict['R_pred'] = pred_tot.T
sio.savemat('cost_accuracy.mat',c_a_dict)
sio.savemat('Weights.mat',weight_dict)
