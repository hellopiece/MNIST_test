import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 生成概率模型
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 正确值
y_ = tf.placeholder("float", [None, 10])

# 计算交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 使用梯度下降算法，步长为0.01最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 开始训练模型1000次
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 模型评估
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))     # 转化成布尔值，再取平均值

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))