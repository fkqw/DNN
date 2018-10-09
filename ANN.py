import tensorflow as tf
import input_data
 
#读取数据
mnist= input_data.read_data_sets('data/',one_hot=True)
trainimg=mnist.train.images
trainimglabel=mnist.train.labels
testimg=mnist.test.images
testlabel=mnist.test.labels
 
print('MNIST loaded')
 
###### 以下是神经网络的结构
 
#网络层，
n_hidden_1=256#
n_hidden_2=128# 隐藏层神经元个数
n_input=784 #输入神经元个数，一共有784个
n_classes=10 #输出
 
#输入和输出
x=tf.placeholder('float',[None,n_input])
y=tf.placeholder('float',[None,n_classes])
 
#网络 参数
 
stddev=0.1
weights={
                                    #第一层 784*256
    'w1':tf.Variable(tf.random_normal([n_input,n_hidden_1],stddev=stddev)),
                                    #第二层 256*128
    'w2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2],stddev=stddev)),
                                    #最后一层 128*10
    'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes],stddev=stddev))
}
biases={
    'b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}
print('NetWork Ready')
 
 
 
#前向传播
#_X data
def multilayer_perceptron(_X,_weights,_biases):
    #_X  _weights['w1'] 相乘,最后加上b1
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(_X,_weights['w1']),_biases['b1']))
    layer_2=tf.nn.sigmoid(tf.add(tf.matmul(layer_1,_weights['w2']),_biases['b2']))
    #第三层，最后一层。最后没有sigmoid，只是输出。
    return (tf.matmul(layer_2,_weights['out'])+_biases['out'])
 
#反向传播
pred=multilayer_perceptron(x,weights,biases)
 
#损失函数
#交叉熵函数损失函数 pred网络预测值，前向传播之，y真实样本
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
#梯度下降
optm=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
corr=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accr=tf.reduce_mean(tf.cast(corr,'float'))
 
init=tf.global_variables_initializer()
print('function ready')
 
###### 以上是神经网络的结构
#超参数定义
trainimg_epochs=20
batch_size=100
disply_step=4
 
#初始化全局变量
sess=tf.Session()
sess.run(init)
#进行迭代
for epoch in  range(trainimg_epochs):
    avg_cost=0
    total_batch=int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        feeds={x:batch_xs,y:batch_ys}
        sess.run(optm,feed_dict=feeds)
        avg_cost+=sess.run(cost,feed_dict=feeds)
    avg_cost=avg_cost/total_batch
    #DISPLAY
    if(epoch+1)% disply_step == 0:
        print('EPOCH:%03d/%03d cost: %.9f' % (epoch,trainimg_epochs,avg_cost))
        feeds={x:batch_xs,y:batch_ys}
        train_acc=sess.run(accr,feed_dict=feeds)
        print('TRAIN ACCURACY:%.3f'%(train_acc))
        feeds={x:mnist.test.images,y:mnist.test.labels}
        test_acc=sess.run(accr,feed_dict=feeds)
        print('TEST ACCURACY:%0.3f'%(test_acc))
 
print('OPTIMIZ FINISHED')
