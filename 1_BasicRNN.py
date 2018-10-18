import numpy as np
import tensorflow as tf

class RNN():
  def __init__(self, batch_size, input_dim, hidden_dim, output_dim):
    self.input = tf.placeholder(tf.float32, [batch_size, None, input_dim])
    self.target = tf.placeholder(tf.float32, [batch_size, None, output_dim])
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
    h = tf.TensorArray(tf.float32, self.seq_len)
    output = tf.TensorArray(tf.float32, self.seq_len)
    time = tf.constant(0, dtype=tf.int32)
    
    # build graph using while_loop
    _loop_cond = lambda time, _1,_2: time<self.seq_len
    final_state_ = tf.while_loop(cond=_loop_cond, body=self._loop_body, loop_vars=(time, h, output))
    
    self.final_state = final_state_
    self.final_output = self.final_state[-1].read(-1)
    
  
  def _loop_body(self, time, h, output):   # what variable should change in each loop? like hidden states and outputs. If there is memory state, that should change as well.
    input_step = self.input_TA.read(time)
    
    def h_prev_0(h):
      return tf.Variable(np.zeros(self.hidden_dim),dtype=tf.float32)
    def h_prev_1(h):
      return h.read(time-1)
    
    h_prev = tf.cond(tf.math.equal(time,0), lambda:h_prev_0(h), lambda:h_prev_1(h))
    h.write(time, tf.sigmoid(tf.matmul(input_step, self.U)+self.b_U+tf.matmul(h_prev, self.W)+self.b_W))
    output.write(time, tf.matmul(h.read(time), self.V)+self.b_V)

    return (time+1, h, output)
    
rnn_1 = RNN(1,20,50,20)
with tf.Session() as sess:
  init = tf.global_variables_initializer()
  sess.run(init)
  
