import scipy.sparse
import tensorflow as tf
import numpy as np
# from tensorflow.python.tools import inspect_checkpoint as ickpt
# ickpt.print_tensors_in_checkpoint_file("./checkPoint111/model.ckpt", tensor_name="", all_tensors=True)
# 初始化矩阵

# input_ur = np.array([1,2,3,4,5,6,7,8,9,10])
#
# aa=tf.not_equal(input_ur, 20)
# print(aa)
# pos_num_r = tf.cast(tf.not_equal(input_ur, 20), 'float32')
# print(pos_num_r)
# print("sss")
# with tf.Session() as sess:
#     print(sess.run(aa))
#     print("..............")
#     print(sess.run(pos_num_r))
# A = np.array([
#     [[1, 2], [3, 4]],
#     [[5, 6], [7, 8]]
# ])
#
# B = np.array([
#     [9, 10],
#     [11, 12],
#     [13, 14]
# ])
# print(A.shape)
# print(B.shape)
# C = np.einsum('ijk,mk->i', A, B)
#
# print(C)
# print(C.shape)

print(64//2)