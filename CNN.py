'''
 卷积神经网络模型
 minist 数据集 n*784  卷积层->filter(3*3*1) 64个filter结果，
                        pooling 2*2 
                        卷积2->filter（3*3*64） 特征128输出
                        pooling 2*2 
                        
                        全链接层 1 特征图 总结 1024
                        全链接2 分类10 
'''
mnist= input_data.read_data_sets('data/',one_hot=True)
trainimg=mnist.train.images
trainimglabel=mnist.train.labels
testimg=mnist.test.images
testlabel=mnist.test.labels

print('MNIST loaded')

n_input=784#像素点
n_output=10# 十分类

weights={
    #卷积层 3,3 filter的大小，1是深度 64 是output特征图 stddev=0.1方差项
    'wc1':tf.Variable(tf.random_normal([3,3,1,64],stddev=0.1)),
    'wc2':tf.Variable(tf.random_normal([3,3,64,128],stddev=0.1)),
    #全链接参数 7*7*128 输入时 转换成1024可以自己定义。前面不能自己定义
    'wd1':tf.Variable(tf.random_normal([7*7*128,1024],stddev=0.1)),
    'wd2':tf.Variable(tf.random_normal([1024,n_output],stddev=0.1))
}
biases={
    'bc1':tf.Variable(tf.random_normal([64],stddev=0.1)),
    'bc2':tf.Variable(tf.random_normal([128],stddev=0.1)),
    'bd1':tf.Variable(tf.random_normal([1024],stddev=0.1)),
    'bd2':tf.Variable(tf.random_normal([n_output],stddev=0.1))
}
#卷积和池化操作,前向传播
# _keepratio保留的比例

def conv_basic(_input,_w,_b,_keepratio):
    #数据预处理 -1 自己推算 ，把 input进行reshape -
    _input_r=tf.reshape(_input,shape=[-1,28,28,1])

#卷积 1 strides=[1,1,1,1]在不同的地方的 大小，w，h和其他 SAME将进行0值填充。
    _conv1=tf.nn.conv2d(_input_r,_w['wc1'],strides=[1,1,1,1],padding='SAME')
    #卷积中激活函数一般选取ReLu
    _conv1=tf.nn.relu(tf.nn.bias_add(_conv1,_b['bc1']))
    #ksize=[window(1),H,W(2*2),]
    _pool1=tf.nn.max_pool(_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    _pool_dr1=tf.nn.dropout(_pool1,_keepratio)
#第二层
    #卷积 1 strides=[1,1,1,1]在不同的地方的 大小，w，h和其他 SAME将进行0值填充。
    _conv2=tf.nn.conv2d(_pool_dr1,_w['wc2'],strides=[1,1,1,1],padding='SAME')
    #卷积中激活函数一般选取ReLu
    _conv2=tf.nn.relu(tf.nn.bias_add(_conv2,_b['bc2']))
    #ksize=[window(1),H,W(2*2),]
    _pool2=tf.nn.max_pool(_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    _pool_dr2=tf.nn.dropout(_pool2,_keepratio)

#全链接层
    # reshape来说明输入时多大。
    _densel=tf.reshape(_pool_dr2,[-1,_w['wd1'].get_shape().as_list()[0]])
    #matmul相乘 然后和偏执项加起来。
    _fc1=tf.nn.relu(tf.add(tf.matmul(_densel,_w['wd1']),_b['bd1']))
    _fc_dr1=tf.nn.dropout(_fc1,_keepratio)
    #第二层全链接层
    _out=tf.add(tf.matmul(_fc_dr1,_w['wd2']),_b['bd2'])
    #
    out={'input':_input,'conv1':_conv1,'pool1':_pool1,'pool1_dr1':_pool_dr1,
         'conv2':_conv2,'pool2':_pool2,'pool2_dr2':_pool_dr2,'dense1':_densel,'fc1':_fc1,           'fc_dr1':_fc_dr1,'out':_out
         }
    return out
print("CNN ready")

#构建迭代运算
#先占位置
x=tf.placeholder(tf.float32,[None,n_input])
y=tf.placeholder(tf.float32,[None,n_output])

keepratio=tf.placeholder(tf.float32)

#FUNCTIONS
_pred=conv_basic(x,weights,biases,keepratio)['out']
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(_pred,y))
optm=tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
_corr=tf.equal(tf.argmax(_pred,1),tf.argmax(y,1))
accr=tf.reduce_mean(tf.cast(_corr,tf.float32))
init=tf.global_variables_initializer()

print('GRAPH READY')

#迭代
sess=tf.Session()
sess.run(init)

training_epochs=15
batch_size=16 # 这里改小方便计算，不然计算量很大
display_step=1

for epoch in  range(training_epochs):
    avg_cost=0
    total_batch=10

    for i in  range(total_batch):
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        sess.run(optm,feed_dict={x:batch_xs,y:batch_ys,keepratio:0.7})
        avg_cost+=sess.run(cost,feed_dict={x:batch_xs,y:batch_ys,keepratio:1})/total_batch

    if epoch % display_step==0:
        print('Epoch:%03d/%03d cost: %.9f'%(epoch,training_epochs,avg_cost))
        train_acc=sess.run(accr,feed_dict={x:batch_xs,y:batch_ys,keepratio:1})
        print('training accuracy:%.3f'%(train_acc))

    print('OPTIMIZATION FINISHED')



