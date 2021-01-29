import tensorflow as tf
import numpy as np
import math
import os
import time
from Utils import preprocess_images

def save_loss(name,err):
    file_name=open(name + '.txt','w')
    file_name.write(str(err))
    file_name.close()

    
def psnr2(target, ref):
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref,dtype=np.float64)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.) )
    return 20*math.log10(255./rmse)
    
class SISR(object):
    
    def __init__(self, sess, image_size=24,label_size=48, batch_size=1,c_dim=1, checkpoint_dir=None, sample_dir=None):
        self.sess = sess
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size
        self.c_dim = c_dim
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.build_model()
    
    def Interpolation(self,y):
        """
        Final z-dim must be equal to hr dimension 48x48
        y = list of 3 images or one image 72x24
        HR dim = 48x48
        LR dim = 24x24
        r=2
        For each little square,one pixel from LR and three from subpixel image
        """
        x = self.images
        x = tf.reshape(self.images,[-1])
        y1=y[0,:,:,0]
        y1 = y1[0:24,0:24]
        y1=tf.reshape(y1,[-1])
        y2=y[0,:,:,1]
        y2 = y2[0:24,0:24]
        y2=tf.reshape(y2,[-1])
        y3=y[0,:,:,2]
        y3 = y3[0:24,0:24]
        y3 = tf.reshape(y3,[-1])
        t2 = tf.constant(np.array([]),dtype = 'float32')
        
        for i in range((y1.get_shape().as_list())[0]):
            t = tf.stack([x[i],y1[i],y2[i],y3[i]],-1)
            t2 = tf.concat([t2,t],-1)
        z=tf.reshape(t2,[48,48])
        
        return z
    
    '''Maxout activation functions variants'''
    
    '''Maxout'''
    def maxout(self,X):
        x1, x2 = tf.split(X, 2, axis=-1)
        x_fin =tf.math.maximum(x1,x2)
        return x_fin#Dimension halved along channels
    
    '''Difference of two maxout'''
    def diff_maxout(self,X):
        x1, x2, x3, x4 = tf.split(X, 4, axis=-1)
        y1 = tf.math.maximum(x1,x2)
        y2 = tf.math.maximum(x3,x4)
        y_fin = tf.math.subtract(y1,y2) 
        return y_fin#Dimension four time smaller then original along channel
    
    '''Minout '''
    def minout_unit(self,X):
        x1, x2 = tf.split(X, 2, axis=-1)
        x_fin = tf.math.minimum(x1,x2)
        return x_fin#Dimension halved along channels
    
    '''Sorting'''
    def sortout_unit(self,X):
        y1 = self.maxout(X)
        y2 = self.minout_unit(X)
        y_fin = tf.concat([y1,y2],axis=-1)
        return y_fin#Same dimension as X input
    
    '''Recursive'''
    def recursive(self,X):
        y = self.maxout(X)
        y2 = self.maxout(y)
        #y3 = self.maxout(y2)
        return y2#Dimension reduced (1/2)^n along channel
    
    
    def log10(self,x):
        numerator = tf.log(x)
        denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator  
        
    def psnr(self,target, ref):

        diff = tf.subtract(ref,target)
        diff =  tf.reshape(diff,[-1])
        rmse = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(diff)))
        fin = tf.multiply(tf.constant(20.),self.log10(tf.div(tf.constant(1.),rmse)));
        return fin
    
    
    
    def build_model(self):
        self.images = tf.placeholder(tf.float32, [None,self.image_size, self.image_size, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [None,self.label_size, self.label_size, self.c_dim], name='labels')
        self.weights = {
          'w1': tf.Variable(tf.random_normal([3, 3, 1, 12], stddev=1e-3), name='w1'),
          'w2': tf.Variable(tf.random_normal([3, 3, 12, 12], stddev=1e-3), name='w2'),
          'w3': tf.Variable(tf.random_normal([3, 3, 12, 12], stddev=1e-3), name='w3'),
          'w4': tf.Variable(tf.random_normal([3, 3, 12, 12], stddev=1e-3), name='w4'),
          'w5': tf.Variable(tf.random_normal([3, 3, 12, 12], stddev=1e-3), name='w5'),
          'w6': tf.Variable(tf.random_normal([3, 3, 12, 3], stddev=1e-3), name='w6')}
        self.biases = {
          'b1': tf.Variable(tf.zeros([12]), name='b1'),
          'b2': tf.Variable(tf.zeros([12]), name='b2'),
          'b3': tf.Variable(tf.zeros([12]), name='b3'),
          'b4': tf.Variable(tf.zeros([12]), name='b4'),
          'b5': tf.Variable(tf.zeros([12]), name='b5'),
          'b6': tf.Variable(tf.zeros([3]), name='b6')}
        
        self.pred = self.model()    
        self.loss = -self.psnr(self.pred,tf.squeeze(self.labels))
        self.psnr_test = self.psnr(self.pred,tf.squeeze(self.labels))
        self.saver = tf.train.Saver(max_to_keep=4)#Salva gli ultimi 4 modelli
        
    def model(self):
        conv1 = self.sortout_unit(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='SAME') + self.biases['b1'])
        conv2 = self.sortout_unit(tf.nn.conv2d(conv1, self.weights['w2'], strides=[1,1,1,1], padding='SAME') + self.biases['b2'])
        conv3 = self.sortout_unit(tf.nn.conv2d(conv2, self.weights['w3'], strides=[1,1,1,1], padding='SAME') + self.biases['b3'])
        conv4 = self.sortout_unit(tf.nn.conv2d(conv3, self.weights['w4'], strides=[1,1,1,1], padding='SAME') + self.biases['b4'])
        conv5 = self.sortout_unit(tf.nn.conv2d(conv4, self.weights['w5'], strides=[1,1,1,1], padding='SAME') + self.biases['b5'])
        conv6 = tf.nn.conv2d(conv5, self.weights['w6'], strides=[1,1,1,1], padding='SAME') + self.biases['b6']
        interpolation= tf.squeeze(self.Interpolation(conv6))
        
        return interpolation
    
    def save(self, checkpoint_dir, step):
        model_name = "SRCNN2.model"
        model_dir = "%s_%s" % ("srcnn", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
#        if not os.path.exists(checkpoint_dir):
#            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s" % ("srcnn", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

        
    
    def train(self, config):
            
        data_dir = '/home/alessio94/Scrivania/Università/Neural_Network'
        train_data, val_data, test_data, train_label, val_label, test_label = preprocess_images(data_dir,config.is_debug)
    
        #Optimization minimizing -psnr index
        self.train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss)
        tf.initialize_all_variables().run()#tf.global_variables_initializer        
        counter = 0
        start_time = time.time()
    
        if self.load(self.checkpoint_dir):
          print(" [*] Load SUCCESS")
        else:
          print(" [!] Load failed...")
        
        if config.is_train:
          if counter > 1800000:#Last 200000 conunter
              config.learning_rate = 0.001
          errore=0
          errore_val=0
          errori=list()
          err_list=list()
          loss_val=[]
          err=0
          print("Training...") 
          last_mean=0
          '''Training loop'''
          for ep in range(config.epoch):
            err_list.append(err)
            if ep % 1000 == 0 :
                last_mean = sum(np.array(err_list[ep-1000:ep]))/1000
            batch_idxs = len(train_data) // config.batch_size
            
            for idx in range(0, batch_idxs):
              batch_images=list()
              batch_labels=list()
              a = np.random.random_integers(0,395,config.batch_size)#Choosing random indices for images
              for elem in a:
                  batch_images.append(train_data[elem])
              batch_images=np.array(batch_images)
              batch_images = np.expand_dims(batch_images,axis = -1)
              for elem in a:
                  batch_labels.append(train_label[elem])
              batch_labels = np.array(batch_labels)
              batch_labels = np.expand_dims(batch_labels,axis=-1)
              counter += 1
              _, err,im , lab = self.sess.run([self.train_op, self.loss,self.pred,self.labels], feed_dict={self.images: batch_images, self.labels: batch_labels})


            print("Epoch: [%2d], step: [%2d], time: [%8s], loss: [%.8f], psnr_test: [%.8f], last_100_mean: [%.8f]"  \
              % ((ep+1), counter, time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time)), (err), errore_val, last_mean))
            errori.append(-err)
            
            if ep % 500 == 0:
                #Each 500 epochs, refresh data taking new images
                train_data, val_data, test_data, train_label, val_label, test_label = preprocess_images(data_dir,config.is_debug)
                self.save(config.checkpoint_dir, counter)
                print('Model saved at epoch:  ' + str(ep))
                errore=0
                #Compute validation error on separated set of images
                for i in range(40):
                    batch_val_images=list()
                    batch_val_labels=list()
                    z = np.random.randint(0,45,config.batch_size)
                    for elem in z:
                        batch_val_images.append(val_data[elem])
                    batch_val_images=np.array(batch_val_images)
                    batch_val_images=np.expand_dims(batch_val_images,axis=-1)
                    for elem in z:
                        batch_val_labels.append(val_label[elem])
                    batch_val_labels=np.array(batch_val_labels)
                    batch_val_labels=np.expand_dims(batch_val_labels,axis=-1)
                    errore2=self.sess.run(self.psnr_test,feed_dict={self.images: batch_val_images, self.labels: batch_val_labels}) 
                    errore +=errore2
                errore_val = errore/40#Error tested on separated set of images
                loss_val.append(errore_val)
                save_loss('loss_validation',loss_val)
                save_loss('loss_error',errori)
                
        else:
              num = 30
              somma_psnr=0
              for num in range(10):
                  batch_val_images=np.array([train_data[num]],dtype=np.float32)
                  batch_val_images=np.expand_dims(batch_val_images,axis=-1)
                  batch_val_labels=np.array([train_label[num]],dtype=np.float32)
                  batch_val_labels=np.expand_dims(batch_val_labels,axis=-1)
                  result = self.pred.eval({self.images: batch_val_images, self.labels: batch_val_labels})
                  result = (255.0 / result.max() * (result - result.min())).astype(np.uint8)
                  result = (result*255).astype(np.uint8)
                  lr_img=train_data[num]
                  lr_img=((lr_img*255) + 0.5).astype(np.uint8)
                  hr_img  = train_label[num]
                  hr_img = (hr_img*255).astype(np.uint8)
                  psnr_img=psnr2(result,hr_img)
                  somma_psnr+=psnr_img
                  print('Immagine: ' + str(num))
              print('Pnsr = ' + str(somma_psnr/10))
          
          
    
        


        
        
        
        