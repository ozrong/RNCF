import argparse

import tensorflow as tf
from MF import Model


if __name__ == '__main__':

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
       saver = tf.train.import_meta_graph('./checkPoint/model.ckpt.meta')
       saver.restore(sess,tf.train.latest_checkpoint('./checkPoint/'))
       graph = tf.get_default_graph()
       userid =graph.get_tensor_by_name("userid:0") #2
       itemid =graph.get_tensor_by_name("itemid:0") #3035
       rate = graph.get_tensor_by_name("rate:0")#4
       drop = graph.get_tensor_by_name("drop:0")

       feed_dict={userid:[2],itemid:[3035],rate:[4],drop:None}


       item_vector = graph.get_tensor_by_name("User_Layer/user_out2:0")
       user_vector = graph.get_tensor_by_name("Item_Layer/item_out2:0")
       # user = tf.get_collection("user_vector")
       # item = tf.get_collection("item_vector")
       # pr = tf.get_collection("prediction_rate")


       # user =  graph.get_tensor_by_name("user_out2:0")
       # item =  graph.get_tensor_by_name("item_out2:0")
       y = graph.get_tensor_by_name("y:0")
       print(sess.run(user_vector,feed_dict))
       print(sess.run(item_vector,feed_dict))
       print(sess.run(y,feed_dict))




