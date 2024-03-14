import tensorflow as tf

from tensorflow.python.client import device_lib

#print(device_lib.list_local_devices())     ## this command list all the processing device GPU and CPU

device_cpu = "/device:CPU:0"
device_gpu = "/device:GPU:0"

#device_name = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
#if device_name[0] == "/device:GPU:0":
#    device_name = "/gpu:0"
    #print('GPU')
#else:
    #print('CPU')
#    device_name = "/cpu:0"


def runDevice(device_name):
	with tf.device(device_name):
		a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
		b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
		c = tf.matmul(a, b)
	with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    		print (sess.run(c))  



print("RUNNING ON CPU (HOPEFULLY):")
runDevice(device_cpu)
print("RUNNING ON G P U :")
runDevice(device_gpu)

