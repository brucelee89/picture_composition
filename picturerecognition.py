from PIL import Image
worldcup = Image.open('D:\\Documents\\worldcup.png')
print(worldcup.mode)
worldcup_rgb = worldcup.convert("RGB")

worldcup_rgb.save('D:\\Documents\\worldcup_rgb.png')
worldcup_rgb.getpixel((0,0))
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

image_filename=['basketball.png','football.png','volleyball.png','badmminton.png','worldcup_rgb.png']

image_filename[4]

image_file = tf.read_file(image_filename[4])

image=tf.image.decode_png(image_file)

image = tf.image.convert_image_dtype(image,dtype=tf.float32)

with tf.Session() as sess:
    pic = sess.run(image)
    
print(pic)

def convert_to(images,labels,name):
   
    num_examples = labels.shape[0]
    if images.shape[0] != num_examples:
        raise ValueError('图片数量%d不匹配标签数%d.'%(images.shape[0], num_examples))
        
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]
    
    filename= os.path.join('./',name + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features = tf.train.Features(feature={
                    'height':_int64_feature(rows),
                    'width':_int64_feature(cols),
                    'depth':_int64_feature(depth),
                    'label':_int64_feature(int(labels[index])),
                    'image_raw':_bytes_feature(image_raw)
                }))
      
        writer.write(example.SerializeToString())
        
plt.imshow(pic)
print(pic.shape)
pic = pic.reshape([1,16,16,3])
print(pic.shape)
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
import os

convert_to(pic,np.array([1]),'worldcup_rgb')

feat={
    'image_raw':tf.FixedLenFeature([],tf.string),
    'label':tf.FixedLenFeature([],tf.int64),
    'height':tf.FixedLenFeature([],tf.int64),
    'width':tf.FixedLenFeature([],tf.int64),
    'depth':tf.FixedLenFeature([],tf.int64)
}

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features=feat)
    image = tf.decode_raw(features['image_raw'],tf.float32)
    label = tf.cast(features['label'],tf.int32)
    height = tf.cast(features['height'],tf.int32)
    width = tf.cast(features['width'],tf.int32)
    depth = tf.cast(features['depth'],tf.int32)
    
    return image, label,  height, width, depth
read_and_decode(tf.train.string_input_producer(['worldcup_rgb.tfrecords']))
def get_all_records(FILE):
    with tf.Session() as sess:
        filename_queue = tf.train.string_input_producer([FILE])
        init_op = (tf.global_variables_initializer(),tf.local_variables_initializer())
        image, label, height, width, depth = read_and_decode(filename_queue)
        image = tf.reshape(image, tf.stack([1,16,16,3]))
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        featuredata = np.reshape(np.zeros(768),(1,16,16,3))
        labeldata=np.reshape(np.zeros(1),(1))
        
        for i in range(1):
            
            example, lab=sess.run([image, label])
            featuredata=np.append(featuredata,example,axis=0)
            lab = np.reshape(np.array(1),1)
            labeldata=np.append(labeldata,lab,axis=0)
        coord.request_stop()
        coord.join(threads)
        return featuredata[1:][:][:][:],labeldata[1:]
data, label=get_all_records('worldcup_rgb.tfrecords')
sess=tf.InteractiveSession()
x=data[0:1]
print('x是输入，形状是：',np.shape(x))

y=np.array([[1,0],[0,1]])

x=tf.constant(x,dtype=tf.float32)
y=tf.constant(y,dtype=tf.float32)

print('y的数据为：\n',y.eval())
keep_prob=0.5
weights= {
    'wc1':tf.Variable(tf.random_normal([2,2,3,2]),dtype=tf.float32),
    'wc2':tf.Variable(tf.random_normal([2,2,2,2]),dtype=tf.float32),

    'wd1':tf.Variable(tf.random_normal([16,2]),dtype=tf.float32),
    'out':tf.Variable(tf.random_normal([2,2]),dtype=tf.float32)
}

biases = {
    'bc1':tf.Variable(tf.random_normal([2]),dtype=tf.float32),
    'bc2':tf.Variable(tf.random_normal([2]),dtype=tf.float32),
    
    'bd1':tf.Variable(tf.random_normal([2]),dtype=tf.float32),
    'out':tf.Variable(tf.random_normal([2]),dtype=tf.float32)
}
sess.run(tf.global_variables_initializer())

print('初始wc1是：',weights['wc1'].eval())
print('初始bc1是：',biases['bc1'].eval())
strides=1
conv1=tf.nn.conv2d(input=x,filter=weights['wc1'],strides=[1,strides,strides,1],padding='SAME')
print('卷积以后的数据和形状为：\n',conv1,'\n',conv1.eval())
conv1 = tf.nn.bias_add(conv1, biases['bc1'])
print('加上偏置以后的值：\n',conv1.eval())
conv1 = tf.nn.relu(conv1)
print('relu激活后的值：\n',conv1.eval())
k=2
pool1=tf.nn.max_pool(conv1,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')
print('池化后的数据：',pool1,'\n',pool1.eval())
strides=1
conv2=tf.nn.conv2d(input=pool1, filter=weights['wc2'],strides=[1,strides,strides,1],padding='SAME')


conv2 = tf.nn.bias_add(conv2,biases['bc2'])
print('二次卷积加偏置后的值：',conv2,'\n',conv2.eval())
conv2=tf.nn.relu(conv2)
print('relu激活后的值：\n',conv2.eval())
k=2
pool2=tf.nn.max_pool(conv2,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')
print('二次池化后的值：',pool2,'\n',pool2.eval())
print('获得权重结构为：',weights['wd1'].get_shape().as_list())
print('\n设置用于reshape的结构为：',[-1,weights['wd1'].get_shape().as_list()[0]])
fc1=tf.reshape(pool2,[-1,weights['wd1'].get_shape().as_list()[0]])
print('\n进入全连接层改变形状成2维的矩阵结构和数据：\n',fc1,'\n',fc1.eval())
fc1=tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
print('\n线性化后的结构和数据：\n',fc1,'\n',fc1.eval())
dropout=0.5
fc1=tf.nn.dropout(fc1,dropout)
print('\n随机dropout后的数据，一部分为0，其他权重翻倍：\n',fc1.eval())
logits = tf.add(tf.matmul(fc1,weights['out']),biases['out'])
conslogits=logits.eval()
print(conslogits)
cent = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y)
print('两条记录的交叉熵',cent.eval())
cost = tf.reduce_mean(cent)
print('cost为交叉熵的均值：',cost.eval())
realy  = np.array([[1,0],
                   [0,1]])
pred_argmax=tf.argmax(logits,1).eval()
print('预测的结果是：\n',realy[pred_argmax])
real_argmax= tf.argmax(y,1).eval()
print('实际结果为:\n',realy[real_argmax])
accur=np.equal(pred_argmax,real_argmax)
print(accur.astype(np.float32).mean())
correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
print(accuracy.eval())
accuracy=accuracy.eval()
print(accuracy)
optimizer.run()
print(cost.eval())
print(accuracy)
