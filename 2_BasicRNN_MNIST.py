import numpy as np
import tensorflow as tf
import os
import time

class RNN():
  def __init__(self, batch_size, input_dim, hidden_dim, output_dim):
    self.input = tf.placeholder(tf.float32, [batch_size, None, input_dim])
    self.target = tf.placeholder(tf.float32, [batch_size, 1, output_dim])
    self.seq_len = tf.placeholder(tf.int32)
    self.hidden_dim = hidden_dim
    
    self.U = tf.get_variable(name='U', shape = [input_dim, hidden_dim], dtype = tf.float32)     # from input to hidden
    self.b_U = tf.get_variable(name='b_U', shape = [hidden_dim], dtype = tf.float32)
    
    self.V = tf.get_variable(name='V', shape = [hidden_dim, output_dim], dtype = tf.float32)      # from hidden to output
    self.b_V = tf.get_variable(name='b_V', shape = [output_dim], dtype = tf.float32)
    
    self.W = tf.get_variable(name='W', shape = [hidden_dim, hidden_dim], dtype = tf.float32)      # from hidden to hidden
    self.b_W = tf.get_variable(name='b_W', shape = [hidden_dim], dtype = tf.float32)
    
    def input_to_TensorArray(value, axis, size=None):
      shape = value.get_shape().as_list()
      rank = len(shape)
      dtype = value.dtype
      array_size = shape[axis] if not shape[axis] is None else size

      if array_size is None:
        raise ValueError("Can't create TensorArray with size None")

      array = tf.TensorArray(dtype=dtype, size=array_size)
      dim_permutation = [axis] + list(range(1, axis)) + [0] + list(range(axis + 1, rank))
      unpack_axis_major_value = tf.transpose(value, dim_permutation)
      full_array = array.unstack(unpack_axis_major_value)

      return full_array
    
    # input data should be converted to TensorArray. However, since input data does not change, it can be declared as self.
    self.input_TA = input_to_TensorArray(self.input, 1, self.seq_len)
    
    # variables that will change in each loop
    h = tf.TensorArray(tf.float32, self.seq_len+1, clear_after_read = False)
    h = h.write(0, tf.constant(np.zeros((1, hidden_dim)), dtype=tf.float32))
    output = tf.TensorArray(tf.float32, self.seq_len)
    time = tf.constant(0, dtype=tf.int32)
    
    # build graph using while_loop
    _loop_cond = lambda time, _1,_2: time<self.seq_len
    final_state_ = tf.while_loop(cond=_loop_cond, body=self._loop_body, loop_vars=(time, h, output))
    
    self.final_state = final_state_
    
    
    self.final_output = self.final_state[-1].read(self.seq_len-1)
    
    #target shape is in accordance with logit shape
    
    loss_ = tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=tf.reshape(self.final_output ,shape=[batch_size, 1, output_dim]))
    self.loss = tf.reduce_mean(loss_)
   
    self.tvars=[self.U, self.b_U, self.V,self.b_V,self.W,self.b_W]
    self.grads = tf.gradients(self.loss, self.tvars)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    self.train_op = optimizer.apply_gradients(zip(self.grads, self.tvars))
#    optimizer = tf.train.AdamOptimizer()
#    self.train_op = optimizer.minimize(self.loss)
    
    #input('press enter to continue')
    
    
  
  def _loop_body(self, time, h, output):   # what variable should change in each loop? like hidden states and outputs. If there is memory state, that should change as well.
    input_step = self.input_TA.read(time)
#    print('input shape:')
#    print(input_step)
#    input("press enter to continue")
    
    h_prev = h.read(time)
    h=h.write(time+1, tf.tanh(tf.matmul(input_step, self.U)+self.b_U+tf.matmul(h_prev, self.W)+self.b_W))
    output=output.write(time, tf.matmul(h.read(time+1), self.V)+self.b_V)

    return (time+1, h, output)
    
def main(argv=None):
  batch_size = 1
  input_dim = 28
  seq_len = 28
  hidden_dim = 30
  output_dim = 10
  n_iter = 600000

  mnist = tf.keras.datasets.mnist

  (x_train, y_train),(x_test, y_test) = mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
 
  rnn = RNN(batch_size,input_dim,hidden_dim,output_dim)
  with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    
    accm_loss=0
    start_time=time.time()
    for i in np.arange(n_iter):
      sample_index = i%50000
      x_batch = x_train[sample_index]
      y_batch_ = y_train[sample_index]
      y_batch = np.zeros(output_dim)
      y_batch[y_batch_]=1
      
      x_batch = x_batch.reshape((batch_size, seq_len, input_dim))
      y_batch = y_batch.reshape((batch_size, 1, output_dim))

      loss,_ = sess.run([rnn.loss, rnn.train_op], feed_dict={rnn.input:x_batch, rnn.target:y_batch, rnn.seq_len:seq_len})
      accm_loss+=loss
      if i%10000==0 and i!=0:
        print('10000 sample average loss:',accm_loss/1.0/10000)
        accm_loss=0
        correct_count = 0
        for j in np.arange(2000):
          x_test_batch = x_test[j].reshape((batch_size, seq_len, input_dim))
          y_test_batch = y_test[j]
      
          preds_, = sess.run([rnn.final_output], feed_dict={rnn.input:x_test_batch, rnn.seq_len:seq_len})
          if y_test_batch==np.argmax(preds_):
            correct_count+=1
        print('iterations:',i,'   test accuracy:',correct_count/1.0/2000)
          #input('press enter to continue')
  print('10 epoch time:',time.time()-start_time, 's')
        
    
        
if __name__ == '__main__':
  tf.app.run()
    
  
