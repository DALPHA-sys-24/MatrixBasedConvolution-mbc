import numpy as np
import tensorflow as tf
from typing import Tuple,List,Dict,Any


def convolution_at(img: np.ndarray,kernel: np.ndarray,i:int,j:int)->float:
    output = 0
    kernel_shape:Tuple =int(kernel.shape[0]),int(kernel.shape[1])
    img_shape: Tuple = int(img.shape[0]), int(img.shape[1])
    center_point:int =int(tf.math.floor((kernel_shape[0]-1)/2))
    height=int(i-center_point)
    width=int(j-center_point)
    for s in range(height,i+center_point+1):
        for r in range(width, j+ center_point+1):
            if (s<0 or s>img_shape[0]-1 or r>img_shape[1]-1 or r<0):continue
            output+=img[s,r]*kernel[s-height,r-width]
    return output

def classical_convolution(img: np.ndarray,kernel: np.ndarray)->np.ndarray:

    img_shape:Tuple =img.shape
    output =np.zeros(shape=img_shape,dtype="float32")
    for i in range(int(img_shape[0])):
        for j in range(int(img_shape[1])):
            output[i,j]=convolution_at(img=img,kernel=kernel,i=i,j=j)
    return output

def shift_(weight:tf.Tensor, strides: int,axis:int=1):
    return  tf.roll(weight, shift=strides, axis=axis)

def build_matrix_padding(input_shape: Tuple[int], pad: int):
    # block
    out_shape: Tuple = input_shape[0]+ 2 * pad, int(input_shape[1]) + 2 * pad
    width_matrix_padding: int = input_shape[0] * input_shape[1]
    height_matrix_padding: int = out_shape[0] * out_shape[1]
    size_block1: Tuple = out_shape[1], width_matrix_padding
    block1 = tf.zeros(shape=size_block1)
    size_block2: Tuple = pad, width_matrix_padding
    block2 = tf.zeros(shape=size_block2)

    line = tf.Variable(np.zeros(shape=(width_matrix_padding)), dtype=tf.float32, trainable=False)
    line[0].assign(1)
    line = tf.reshape(line, shape=(1, line.shape[0]))
    # initialisation
    M = line
    new_line = line
    matrix = block1

    for i in range(1, out_shape[0] - 1):
        matrix = tf.concat([matrix, block2], 0)

        for j in range(1, input_shape[1]):
            new_line = shift_(new_line, 1)
            M = tf.concat([M, new_line], 0)
        matrix = tf.concat([matrix, M], 0)
        matrix = tf.concat([matrix, block2], 0)
        del M
        new_line = shift_(new_line, 1)
        M = new_line

    matrix = tf.concat([matrix, block1], 0)
    assert matrix.shape == (height_matrix_padding, width_matrix_padding)
    return matrix


