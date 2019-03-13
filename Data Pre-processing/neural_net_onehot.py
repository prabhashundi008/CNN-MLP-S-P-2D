#%% Neural Networks
import numpy as np
import tensorflow as tf

def array_it(filename):
    a = open(filename,'r')
    b = a.read()
    b = b.split('\n')
    for i in range(len(b)):
        b[i] = b[i].split('\t')
    del b[-1]
    for i in range(len(b)):
        for j in range(len(b[i])):
            b[i][j] = float(b[i][j])
    c = np.array(b)
    return c

def new_weights(shape):
#    return tf.Variable(tf.constant(0.0,shape = shape))
    return tf.Variable(tf.truncated_normal(shape,stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05,shape = [length]))

#x_train = array_it('Xsi_train.txt')
x_train = array_it('X_train.txt')
y1_train = array_it('Y_train.txt')
#y2_train = array_it('Y2_train.txt')
#x_test = array_it('Xsi_test.txt')
x_test = array_it('X_test.txt')
y1_test = array_it('Y_test.txt')
x_test2 = array_it('X_test2.txt')
y1_test2 = array_it('Y_test2.txt')
#y2_test = array_it('Y2_test.txt')
#x_test = array_it('p_validate.txt')
#y_test = array_it('y_validate.txt')
n1 = 29; n2 = 25; n3 = 20; n4 = 15; n5 = 8; n6 = 3;# n7 = 8;# n8 = 3;
#n1 = 2; n2 = 3; n3 = 10; n4 = 15; n5 = 10; n6 = 3;
x = tf.placeholder(dtype=tf.float32,shape=(None,n1))
y1_ = tf.placeholder(dtype=tf.float32,shape=(None,1))
#y2_ = tf.placeholder(dtype=tf.float32,shape=(None,10))
#y1_cls = tf.argmax(y1_,dimension=1)
#y2_cls = tf.argmax(y2_,dimension=1)

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
tr_cost = []
va_cost = []
cost1 = tf.square(y1-y1_)
#cost1 = tf.nn.softmax_cross_entropy_with_logits(labels=y1_,logits=y1)
#cost2 = tf.nn.softmax_cross_entropy_with_logits(labels=y2_,logits=y2)
cost = tf.reduce_mean(cost1)
train_step = tf.train.AdamOptimizer(1e-2).minimize(cost)
#y1cls = tf.argmax(y1,dimension=1)
#y2cls = tf.argmax(y2,dimension=1)
#tf.summary.scalar('Loss', cost)
#summary_op = tf.summary.merge_all()
#saver = tf.train.Saver()
#result_dir = './resultsb/' # directory where the results from the training are saved
#%%
sess = tf.InteractiveSession()
#sess = tf.Session()
#summary_writer = tf.summary.FileWriter(result_dir, sess.graph)
sess.run(tf.global_variables_initializer())
for i in range(9):
    print(sess.run(cost,{x:x_train,y1_:y1_train}))
    print(sess.run(cost,{x:x_test,y1_:y1_test}))
    sess.run(train_step,{x:x_train,y1_:y1_train})
print(sess.run(cost,{x:x_train,y1_:y1_train}))
test_cost = sess.run(cost,{x:x_test,y1_:y1_test})
print(test_cost)

for i in range(25000):
    sess.run(train_step,{x:x_train,y1_:y1_train})
    if i%50 == 0:
        train_cost1 = sess.run(cost,{x:x_train,y1_:y1_train})
        tr_cost.append(train_cost1)
        print('Steps ',i,' Train Cost ',train_cost1)
        test_cost1 = sess.run(cost,{x:x_test,y1_:y1_test})
        va_cost.append(test_cost1)
        print('Steps ',i,' Test Cost ',test_cost1)
        if test_cost1 > test_cost:
            break
        test_cost = test_cost1
#        summary_str = sess.run(summary_op, feed_dict={x:x_test,y1_:y1_test})
#        summary_writer.add_summary(summary_str, i)
#        summary_writer.flush()
        
print(sess.run(cost,{x:x_test2,y1_:y1_test2}))   

import matplotlib.pyplot as plt
#%%
true_ans1 = sess.run(y1_,{x:x_test2,y1_:y1_test2})
#true_ans2 = sess.run(y2_cls,{x:x_test,y1_:y1_test,y2_:y2_test})
pred_ans1 = sess.run(y1,{x:x_test2,y1_:y1_test2})
#pred_ans2 = sess.run(y2cls,{x:x_test,y1_:y1_test,y2_:y2_test})
plt.plot(true_ans1,pred_ans1,'ro')
#  plt.plot(pred_ans2,true_ans2,'ro')
print(np.mean(np.abs(true_ans1-pred_ans1))) 
true_mean = np.mean(true_ans1)
M = true_ans1 - true_mean*np.ones((true_ans1.size,1))
E = true_ans1 - pred_ans1
R2 = 1 - (np.sum(E*E))/(np.sum(M*M))
print(R2)

#print(np.mean(np.abs(true_ans2-pred_ans2))) 
 
summary = np.array([(true_ans1[i][0],pred_ans1[i][0]) for i in range(len(true_ans1))])
np.savetxt('results_te.txt',summary,fmt='%s',delimiter='\t')
#%%
true_ans1 = sess.run(y1_,{x:x_test,y1_:y1_test})
#true_ans2 = sess.run(y2_cls,{x:x_test,y1_:y1_test,y2_:y2_test})
pred_ans1 = sess.run(y1,{x:x_test,y1_:y1_test})
#pred_ans2 = sess.run(y2cls,{x:x_test,y1_:y1_test,y2_:y2_test})
plt.plot(true_ans1,pred_ans1,'ro')
#  plt.plot(pred_ans2,true_ans2,'ro')
print(np.mean(np.abs(true_ans1-pred_ans1))) 
true_mean = np.mean(true_ans1)
M = true_ans1 - true_mean*np.ones((true_ans1.size,1))
E = true_ans1 - pred_ans1
R2 = 1 - (np.sum(E*E))/(np.sum(M*M))
print(R2)

#print(np.mean(np.abs(true_ans2-pred_ans2))) 
 
summary = np.array([(true_ans1[i][0],pred_ans1[i][0]) for i in range(len(true_ans1))])
np.savetxt('results_va.txt',summary,fmt='%s',delimiter='\t')

true_ans1 = sess.run(y1_,{x:x_train,y1_:y1_train})
#true_ans2 = sess.run(y2_cls,{x:x_test,y1_:y1_test,y2_:y2_test})
pred_ans1 = sess.run(y1,{x:x_train,y1_:y1_train})
#pred_ans2 = sess.run(y2cls,{x:x_test,y1_:y1_test,y2_:y2_test})
plt.plot(true_ans1,pred_ans1,'ro')
#  plt.plot(pred_ans2,true_ans2,'ro')
print(np.mean(np.abs(true_ans1-pred_ans1))) 
true_mean = np.mean(true_ans1)
M = true_ans1 - true_mean*np.ones((true_ans1.size,1))
E = true_ans1 - pred_ans1
R2 = 1 - (np.sum(E*E))/(np.sum(M*M))
print(R2)

#print(np.mean(np.abs(true_ans2-pred_ans2))) 
 
summary = np.array([(true_ans1[i][0],pred_ans1[i][0]) for i in range(len(true_ans1))])
np.savetxt('results_tr.txt',summary,fmt='%s',delimiter='\t')
#%%
import matplotlib.pyplot as plt

true_ans1 = sess.run(y1_,{x:x_test2,y1_:y1_test2})
#true_ans2 = sess.run(y2_cls,{x:x_test,y1_:y1_test,y2_:y2_test})
pred_ans1 = sess.run(y1,{x:x_test2,y1_:y1_test2})
#pred_ans2 = sess.run(y2cls,{x:x_test,y1_:y1_test,y2_:y2_test})
plt.plot(true_ans1,pred_ans1,'ro')
#  plt.plot(pred_ans2,true_ans2,'ro')
print(np.mean(np.abs(true_ans1-pred_ans1))) 
true_mean = np.mean(true_ans1)
M = true_ans1 - true_mean*np.ones((true_ans1.size,1))
E = true_ans1 - pred_ans1
R2 = 1 - (np.sum(E*E))/(np.sum(M*M))
print(R2)

#print(np.mean(np.abs(true_ans2-pred_ans2))) 
 
summary = np.array([(true_ans1[i][0],pred_ans1[i][0]) for i in range(len(true_ans1))])
np.savetxt('results.txt',summary,fmt='%s',delimiter='\t')
#%% Auto-encoder compression

import numpy as np
import tensorflow as tf

def array_it(filename):
    a = open(filename,'r')
    b = a.read()
    b = b.split('\n')
    for i in range(len(b)):
        b[i] = b[i].split('\t')
    del b[-1]
    for i in range(len(b)):
        for j in range(len(b[i])):
            b[i][j] = float(b[i][j])
    c = np.array(b)
    return c

def new_weights(shape):
#    return tf.Variable(tf.constant(0.0,shape = shape))
    return tf.Variable(tf.truncated_normal(shape,stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05,shape = [length]))

x_train = array_it('x_norm_train.txt')
#y_train = array_it('y_train.txt')
x_test = array_it('x_norm_test.txt')
#y_test = array_it('y_test.txt')
#x_test = array_it('p_validate.txt')
#y_test = array_it('y_validate.txt')

x = tf.placeholder(dtype=tf.float32,shape=(None,243))
#%%
W1 = new_weights((243,100))
B1 = new_biases(100)

#W2 = new_weights((100,50))
#B2 = new_biases(50)
#

#W3 = new_weights((50,20))
#B3 = new_biases(20)
#
#W4 = new_weights((20,50))
#B4 = new_biases(50)

#W4 = new_weights((100,100))
#B4 = new_biases(100)

W5 = new_weights((100,100))
B5 = new_biases(100)

W6 = new_weights((100,243))
B6 = new_biases(243)

layer1 = tf.matmul(x,W1)+B1
#layer1 = tf.nn.tanh(layer1)
#layer2 = tf.matmul(layer1,W2)+B2
#layer2 = tf.nn.tanh(layer2)
#layer3 = tf.matmul(layer2,W3)+B3
#layer3 = tf.nn.tanh(layer3)
#layer4 = tf.matmul(layer3,W4)+B4
#layer4 = tf.nn.relu(layer4)
layer5 = tf.matmul(layer1,W5)+B5
#layer5 = tf.nn.relu(layer5)
y = tf.matmul(layer5,W6)+B6
#y = tf.nn.sigmoid(y)
#%%
cost = tf.reduce_sum(tf.square(y-x))
train_step = tf.train.AdamOptimizer(2e-3).minimize(cost)
#%%
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(cost,{x:x_train}))
print(sess.run(cost,{x:x_train}))
sess.run(train_step,{x:x_train})
print(sess.run(cost,{x:x_train}))
print(sess.run(cost,{x:x_train}))
sess.run(train_step,{x:x_train})
print(sess.run(cost,{x:x_train}))
print(sess.run(cost,{x:x_train}))
sess.run(train_step,{x:x_train})
print(sess.run(cost,{x:x_train}))
print(sess.run(cost,{x:x_train}))
sess.run(train_step,{x:x_train})
print(sess.run(cost,{x:x_train}))
print(sess.run(cost,{x:x_train}))
sess.run(train_step,{x:x_train})
print(sess.run(cost,{x:x_train}))
print(sess.run(cost,{x:x_train}))
sess.run(train_step,{x:x_train})
print(sess.run(cost,{x:x_train}))
print(sess.run(cost,{x:x_train}))
sess.run(train_step,{x:x_train})
print(sess.run(cost,{x:x_train}))
print(sess.run(cost,{x:x_train}))
sess.run(train_step,{x:x_train})
print(sess.run(cost,{x:x_train}))
print(sess.run(cost,{x:x_train}))
sess.run(train_step,{x:x_train})
print(sess.run(cost,{x:x_train}))
test_cost = sess.run(cost,{x:x_train})
print(test_cost)
#%%
for i in range(50000):
    sess.run(train_step,{x:x_train})
    if i%100 == 0:
        print('Steps ',i,' Train Cost ',sess.run(cost,{x:x_train}))
        test_cost1 = sess.run(cost,{x:x_test})
        print('Steps ',i,' Test Cost ',test_cost1)
        if test_cost1 > test_cost:
            break
        test_cost = test_cost1
    
#%%
comp_train = sess.run(layer5,{x:x_train})
comp_test = sess.run(layer5,{x:x_test})
#%%
x_train = comp_train
y_train = array_it('y_train.txt')
#x_test = array_it('Xsi_test.txt')
x_test = comp_test
y_test = array_it('y_test.txt')
#x_test = array_it('p_validate.txt')
#y_test = array_it('y_validate.txt')

x = tf.placeholder(dtype=tf.float32,shape=(None,100))
y_ = tf.placeholder(dtype=tf.float32,shape=(None,1))
#%%
W1 = new_weights((100,20))
B1 = new_biases(20)

W2 = new_weights((20,5))
B2 = new_biases(5)
#
#W3 = new_weights((50,10))
#B3 = new_biases(10)
##
#W4 = new_weights((10,4))
#B4 = new_biases(4)
#
#W5 = new_weights((150,30))
#B5 = new_biases(30)

W6 = new_weights((5,1))
B6 = new_biases(1)

layer1 = tf.matmul(x,W1)+B1
layer1 = tf.nn.relu(layer1)
layer2 = tf.matmul(layer1,W2)+B2
layer2 = tf.nn.relu(layer2)
#layer3 = tf.matmul(layer2,W3)+B3
#layer3 = tf.nn.relu(layer3)
#layer4 = tf.matmul(layer3,W4)+B4
#layer4 = tf.nn.relu(layer4)
#layer5 = tf.matmul(layer4,W5)+B5
#layer5 = tf.nn.relu(layer5)
y = tf.matmul(layer2,W6)+B6
y = tf.nn.relu(y)
#%%
cost = tf.reduce_sum(tf.square(y-y_))
train_step = tf.train.AdamOptimizer(2e-3).minimize(cost)
#%%
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(cost,{x:x_train,y_:y_train}))
print(sess.run(cost,{x:x_test,y_:y_test}))
sess.run(train_step,{x:x_train,y_:y_train})
print(sess.run(cost,{x:x_train,y_:y_train}))
print(sess.run(cost,{x:x_test,y_:y_test}))
sess.run(train_step,{x:x_train,y_:y_train})
print(sess.run(cost,{x:x_train,y_:y_train}))
print(sess.run(cost,{x:x_test,y_:y_test}))
sess.run(train_step,{x:x_train,y_:y_train})
print(sess.run(cost,{x:x_train,y_:y_train}))
print(sess.run(cost,{x:x_test,y_:y_test}))
sess.run(train_step,{x:x_train,y_:y_train})
print(sess.run(cost,{x:x_train,y_:y_train}))
print(sess.run(cost,{x:x_test,y_:y_test}))
sess.run(train_step,{x:x_train,y_:y_train})
print(sess.run(cost,{x:x_train,y_:y_train}))
print(sess.run(cost,{x:x_test,y_:y_test}))
sess.run(train_step,{x:x_train,y_:y_train})
print(sess.run(cost,{x:x_train,y_:y_train}))
print(sess.run(cost,{x:x_test,y_:y_test}))
sess.run(train_step,{x:x_train,y_:y_train})
print(sess.run(cost,{x:x_train,y_:y_train}))
print(sess.run(cost,{x:x_test,y_:y_test}))
sess.run(train_step,{x:x_train,y_:y_train})
print(sess.run(cost,{x:x_train,y_:y_train}))
print(sess.run(cost,{x:x_test,y_:y_test}))
sess.run(train_step,{x:x_train,y_:y_train})
print(sess.run(cost,{x:x_train,y_:y_train}))
test_cost = sess.run(cost,{x:x_test,y_:y_test})
print(test_cost)
#%%
for i in range(50000):
    sess.run(train_step,{x:x_train,y_:y_train})
    if i%100 == 0:
        print('Steps ',i,' Train Cost ',sess.run(cost,{x:x_train,y_:y_train}))
        test_cost1 = sess.run(cost,{x:x_test,y_:y_test})
        print('Steps ',i,' Test Cost ',test_cost1)
        if test_cost1 > test_cost:
            break
        test_cost = test_cost1
    
#%%
import matplotlib.pyplot as plt
true_ans = sess.run(y_,{x:x_test,y_:y_test})
pred_ans = sess.run(y,{x:x_test,y_:y_test})
plt.plot(pred_ans,true_ans,'ro')
print(np.mean(np.abs(true_ans-pred_ans)))

summary = np.array([(true_ans[i][0],pred_ans[i][0]) for i in range(len(true_ans))])
np.savetxt('nn_train.txt',summary,fmt='%s',delimiter='\t')