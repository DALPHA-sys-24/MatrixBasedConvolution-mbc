from .utils import build_matrix_padding

import numpy as np
import tensorflow as tf
from typing import Tuple,List
from tensorflow.keras import activations


class matrix_conv_1d(object):
    def __init__(self,  kernel_size=3,
                        stride=1,
                        padding='valid',
                        use_phi=True,
                        activation=None,
                        use_lambda_out=False,
                        use_lambda_in=False,):


        
        self.stride = stride
        self.use_phi = use_phi
        self.padding = padding
        self.kernel_size = kernel_size
        self.use_lambda_in = use_lambda_in
        self.use_lambda_out = use_lambda_out
        self.activation = activations.get(activation)

    
    def build(self, input_shape:Tuple)->None:
        
        if self.padding not in ['valid', 'same']:
            raise NotImplemented("Padding must be 'valid' or 'same'")  
        
        if self.stride < 1:
            raise ValueError("self.stride must be >= 1")
        
        if self.stride >1 and self.padding=="same":
            raise ValueError("Not implemented: self.padding='same' and self.stride>1. If self.padding='same', self.stride=1")  
        
        if self.kernel_size <= 1:
            raise ValueError("self.kernel_size must be >= 1")
        if self.kernel_size% 2 == 0:
            raise ValueError("self.kernel_size must be odd")
            
        if self.padding == "same" and self.stride==1:
            self.P = int(tf.math.floor((self.kernel_size-1) / 2))
            self.input_shape=input_shape[1]+2*self.P
            self.output_shape: int = input_shape[1]
            self.pad=tf.pad(tf.linalg.diag(tf.ones((input_shape[1],),dtype=tf.float32)), paddings=[[self.P , self.P ], [0, 0]], mode='CONSTANT', constant_values=0)
            self.indices_phi()
        
        elif self.padding == "valid" and self.stride>=1:
            self.P =0
            self.input_shape=input_shape[1]
            self.output_shape: int = int(tf.math.floor((self.input_shape + 2*self.P- self.kernel_size) / self.stride) + 1)
            self.pad=tf.linalg.diag(tf.ones((input_shape[1],),dtype=tf.float32))
            self.indices_phi()
        else:
            raise NotImplemented("Not implemented !")
            
        # Ensure that the number of indices matches the expected size for the convolution operation
        assert len(self.indices) == self.output_shape * self.kernel_size
        assert len(self.indices_in) == self.input_shape
        assert len(self.indices_out) == self.output_shape
        
        # \kernel phi
        if self.use_phi:
            self.kernel = tf.random.uniform(shape=(self.kernel_size,),minval=-0.5,maxval=0.5)
            
        else:
            self.kernel = tf.ones((self.kernel_size,),dtype=tf.float32)
        
        # \phi
        self.phi = tf.sparse.to_dense(tf.sparse.SparseTensor(indices=self.indices,
                                                             values= tf.reshape(tf.tile(tf.expand_dims(self.kernel, axis=0), [self.output_shape, 1]),
                                                             shape=(self.output_shape * self.kernel_size,)),
                                                             dense_shape=(self.output_shape,self.input_shape)))
            
        # \lambda_in
        if self.use_lambda_in:
            lambda_in = tf.random.uniform(shape=(self.input_shape,),minval=-0.5,maxval=0.5)
        else:
            lambda_in = tf.ones((self.input_shape,),dtype=tf.float32)
            
        self.lambda_in =  tf.sparse.to_dense(tf.sparse.SparseTensor(indices=self.indices_in, values= lambda_in,dense_shape=(self.input_shape,self.input_shape)))

        # \lambda_out
        if self.use_lambda_out:
            lambda_out = tf.random.uniform(shape=(self.output_shape,),minval=-0.5,maxval=0.5)
        else:
            lambda_out = tf.zeros((self.output_shape,),dtype=tf.float32)
            
        self.lambda_out = tf.sparse.to_dense(tf.sparse.SparseTensor(indices=self.indices_out, values=lambda_out,dense_shape=(self.output_shape,self.output_shape)))
        
        # encode = phi @ lambda_in
        encode = tf.matmul(self.phi, self.lambda_in)

        # decode = phi^T @ lambda_out
        decode = tf.matmul( self.lambda_out,self.phi)
        
        # kernel = encode - decode
        self.noyau = encode - decode
        self.custom = True
        
    
    def conv(self, inputs: tf.Tensor):
        if not self.custom:
            raise ValueError("The layer has not been built yet.")



        # apply padding then matmul
        outputs = tf.matmul(self.pad, inputs, transpose_b=True)  
        outputs = tf.matmul(self.noyau, outputs)                     
        outputs = tf.transpose(outputs)                          

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs
    
    @tf.function(jit_compile=True)
    def conv_jit(self, inputs: tf.Tensor):
         return self.conv(inputs)

    def indices_phi(self,*args):
        self.indices: List[Tuple] = list()
        self.indices_in: List[Tuple] = list()
        self.indices_out: List[Tuple] = list()
        for i in range(self.output_shape):
            self.indices_out.append((i,i))
            for j in range(i*self.stride,i*self.stride+self.kernel_size):
                self.indices.append((i,j))        
        for i in range(self.input_shape):
            self.indices_in.append((i,i))


class matrix_conv_2d(object):
    def __init__(self,kernel_size=3,
                      stride=1,
                      padding='valid',
                      use_phi=True,
                      activation=None,
                      use_lambda_out=False,
                      use_lambda_in=False):



        self.use_phi = use_phi
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.use_lambda_out = use_lambda_out
        self.use_lambda_in = use_lambda_in
        self.activation = activations.get(activation)

    def build(self, input_shape):
        
        if self.padding not in {"same", "valid"}:
            raise ValueError("Padding must be 'same' or 'valid'.")
        if self.stride < 1:
            raise ValueError("stride must be >= 1.")
        if self.stride > 1 and self.padding == "SAME":
            raise ValueError("Not implemented: padding='SAME' and stride>1. If padding='SAME', stride=1.")
        if self.kernel_size <= 1:
            raise ValueError("Kernel size must be >= 1.")
        if self.kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")
        if not isinstance(input_shape, tf.TensorShape):
            if not isinstance(input_shape, (list, tuple)):
                raise ValueError("Input shape must be a tf.TensorShape or a tuple/list.")
        
        # -----------------------------------------matrix_pad------------------------------------
        if self.padding == "same":
            self.P = int(tf.math.floor((self.kernel_size - 1) / 2))
            self.input_shape: Tuple = input_shape[1] + 2 * self.P, input_shape[2] + 2 * self.P
            self.pad = build_matrix_padding(input_shape=(input_shape[1], input_shape[2]), pad=self.P)
            self.indices_phi()
        elif self.padding == "valid":
            self.input_shape: Tuple = input_shape[1], input_shape[2]
            self.pad = tf.constant(np.identity(input_shape[1] * input_shape[2]), dtype=tf.float32)
            self.indices_phi()
        else:
            raise ValueError("Padding not found. Use 'same' or 'valid'.")

        # print("Input shape after padding:", self.input_shape)
        

        # \kernel phi
        if self.use_phi:
            self.kernel = tf.random.uniform(shape=(self.kernel_size*self.kernel_size,),minval=-0.5,maxval=0.5) 
        else:
            self.kernel = tf.ones((self.kernel_size*self.kernel_size,),dtype=tf.float32)

        # \phi
        self.phi = tf.sparse.to_dense(tf.sparse.SparseTensor(indices=self.indices,
                                                             values= tf.reshape(tf.tile(tf.expand_dims(self.kernel, axis=0), [int(self.output_shape[0]*self.output_shape[1]), 1]),
                                                             shape=(int(self.output_shape[0]*self.output_shape[1] * self.kernel_size*self.kernel_size),)),
                                                             dense_shape=(int(self.output_shape[0]*self.output_shape[1]),int(self.input_shape[0]*self.input_shape[1]))))
        # \lambda_in
        if self.use_lambda_in:
            lambda_in = tf.random.uniform(shape=(self.input_shape[0] * self.input_shape[1],),minval=-0.5,maxval=0.5)
        else:
            lambda_in = tf.ones((self.input_shape[0] * self.input_shape[1],),dtype=tf.float32)
            
        self.lambda_in =  tf.sparse.to_dense(tf.sparse.SparseTensor(indices=self.indices_in,
                                                                    values= lambda_in,
                                                                    dense_shape=(self.input_shape[0]*self.input_shape[1],self.input_shape[0]*self.input_shape[1])))

        # \lambda_out
        if self.use_lambda_out:
            lambda_out = tf.random.uniform(shape=(self.output_shape[0]*self.output_shape[1],),minval=-0.5,maxval=0.5)
        else:
            lambda_out = tf.zeros((self.output_shape[0]*self.output_shape[1],),dtype=tf.float32)
            
        self.lambda_out = tf.sparse.to_dense(tf.sparse.SparseTensor(indices=self.indices_out,
                                                                    values=lambda_out,
                                                                    dense_shape=(self.output_shape[0]*self.output_shape[1],self.output_shape[0]*self.output_shape[1])))
        # encode = phi @ lambda_in
        encode = tf.matmul(self.phi, self.lambda_in)

        # decode = phi^T @ lambda_out
        decode = tf.matmul( self.lambda_out,self.phi)
    
        # noyau = encode - decode
        self.noyau = encode - decode
            
    
        # ---------------------------------------------------------------------------------------
        self.custom = True

    def conv(self,  inputs: tf.Tensor):
        if not self.custom:
            raise ValueError("The layer has not been built yet.")
        # ----------------------------------Inputs---------------------------------------------
  
        flatten = tf.reshape(inputs, shape=(-1, inputs.shape[1] * inputs.shape[2], inputs.shape[3]))
        upFlatten = tf.matmul(a=self.pad, b=flatten)
        upFlatten=tf.transpose(upFlatten, perm=[2, 0, 1])
        
        # -----------------------------------Outputs-------------------------------------------------
        outputs = tf.matmul(a=upFlatten, b=self.noyau , transpose_b=True)
        outputs = tf.transpose(outputs, perm=[1, 2, 0])
        outputs = tf.reshape(outputs, shape=(-1, self.output_shape[0], self.output_shape[1], inputs.shape[3]))

        if self.activation is not None:
            outputs = self.activation(outputs)
        else:
            pass
        return outputs    
    
    @tf.function(jit_compile=True)
    def conv_jit(self, inputs: tf.Tensor):
         return self.conv(inputs)

    
    def indices_phi(self, *args):
        self.indices: List[Tuple] = list()
        self.indices_in: List[Tuple] = list()
        self.indices_out: List[Tuple] = list()
        self.output_shape: Tuple = int(tf.math.floor((self.input_shape[0] - self.kernel_size) / self.stride) + 1), int(tf.math.floor((self.input_shape[1] - self.kernel_size) / self.stride) + 1)

        for j in range(int(self.input_shape[1] * self.input_shape[0])):
            self.indices_in.append((j,j))
        
        for i in range(int(self.output_shape[1] * self.output_shape[0])):
            self.indices_out.append((i,i))
        
        count: int = 1
        shift: int = 0
        for i in range(int(self.output_shape[0] * self.output_shape[1])):
            if i == count * self.output_shape[1]:
                count += 1
                shift += self.kernel_size + (self.stride - 1) * self.input_shape[1]
            else:
                if shift:
                    shift += self.stride
                else:
                    shift += 1
            for block in range(self.kernel_size):
                for j in range(self.kernel_size):
                    self.indices.append((i, block * self.input_shape[1] + shift - 1 + j))
    
