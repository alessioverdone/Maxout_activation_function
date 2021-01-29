''' Here I specified the type of model to train '''
from Diff_of_maxout_model import SISR 
import tensorflow as tf
import os

flags = tf.app.flags
flags.DEFINE_integer("epoch", 5000, "Number of epoch [15000]")
flags.DEFINE_integer("batch_size", 2, "The size of batch images [128]")
flags.DEFINE_integer("image_size", 24, "The size of image to use [33]")
flags.DEFINE_integer("label_size", 48, "The size of label to produce [21]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint3]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample3]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
flags.DEFINE_boolean("is_debug", False, "True for debugging, False for testing or whole image set [True]")
FLAGS = flags.FLAGS

def main(_):

  if not os.path.exists(FLAGS.checkpoint_dir):#Se non esiste il path,crealo
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):#Se non esiste il path,crealo
    os.makedirs(FLAGS.sample_dir)
  print('Start program...')
  with tf.Session() as sess:
    sisr = SISR(sess, 
                  image_size=FLAGS.image_size, 
                  label_size=FLAGS.label_size, 
                  batch_size=FLAGS.batch_size,
                  c_dim=FLAGS.c_dim, 
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  sample_dir=FLAGS.sample_dir)

    sisr.train(FLAGS)
    
if __name__ == '__main__':
  tf.app.run()
