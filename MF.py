import tensorflow as tf
import numpy as np
import argparse
from dataset import DataSet
import sys
import os
import heapq
import math



class Model:
    def __init__(self, args):
        self.dataName = args.dataName
        self.dataSet = DataSet(self.dataName)
        self.shape = self.dataSet.shape
        self.maxRate = self.dataSet.maxRate
        self.train = self.dataSet.train
        self.test = self.dataSet.test
        self.negNum = args.negNum
        #############
        self.initializer = args.initializer
        self.activation_func = args.activation
        self.regularizer_rate=args.regularizer
        self.inference()
        self.dropout = args.dropout
        self.embed_size = args.embed_size
        #############
#        self.testNeg = self.dataSet.getTestNeg(self.test, 99)
        self.maxEpochs = args.maxEpochs
        self.batchSize = args.batchSize
        self.topK = args.topK
        self.earlyStop = args.earlyStop
        self.add_embedding_matrix()

        self.add_placeholders()
        self.add_model()
        self.add_loss()
        self.lr = args.lr
        self.add_train_step()
        self.checkPoint = args.checkPoint
        self.init_sess()

    def inference(self):
        """ Initialize important settings """
        self.regularizer = tf.contrib.layers.l2_regularizer(self.regularizer_rate)

        if self.initializer == 'Normal':
            self.initializer = tf.truncated_normal_initializer(stddev=0.01)
        elif self.initializer == 'Xavier_Normal':
            self.initializer = tf.contrib.layers.xavier_initializer()
        else:
            self.initializer = tf.glorot_uniform_initializer()

        if self.activation_func == 'ReLU':
            self.activation_func = tf.nn.relu
        elif self.activation_func == 'Leaky_ReLU':
            self.activation_func = tf.nn.leaky_relu
        elif self.activation_func == 'ELU':
            self.activation_func = tf.nn.elu



    def add_placeholders(self):
        self.user = tf.placeholder(shape=(None,),dtype=tf.int32,name="userid")
        self.item = tf.placeholder(shape=(None,),dtype=tf.int32,name="itemid")
        self.rate = tf.placeholder(shape=(None,),dtype=tf.float32,name='rate')
        self.drop = tf.placeholder(tf.float32,name="drop")

    def add_embedding_matrix(self):
        self.user_Embedding = tf.Variable(tf.truncated_normal(shape=[self.shape[0], self.embed_size], dtype=tf.float32, mean=0.0,stddev=0.01), name="user_Embedding")
        self.item_Embedding = tf.Variable(tf.truncated_normal(shape=[self.shape[1], self.embed_size ], dtype=tf.float32, mean=0.0,stddev=0.01), name="item_Embedding")

    def add_model(self):
        self.user_input = tf.nn.embedding_lookup(self.user_Embedding, self.user)
        self.item_input = tf.nn.embedding_lookup(self.item_Embedding, self.item)

        with tf.name_scope("MNN"):
            self.interaction = tf.concat([self.user_input, self.item_input],axis=-1, name='interaction')

            self.layer1_MLP = tf.layers.dense(inputs=self.interaction,
                                              units=self.embed_size,
                                              activation=self.activation_func,
                                              kernel_initializer=self.initializer,
                                              kernel_regularizer=self.regularizer,
                                              name='layer1_MLP')
            self.layer1_MLP = tf.layers.dropout(self.layer1_MLP, rate=self.dropout)

            self.layer2_MLP = tf.layers.dense(inputs=self.layer1_MLP,
                                              units=self.embed_size //2,
                                              activation=self.activation_func,
                                              kernel_initializer=self.initializer,
                                              kernel_regularizer=self.regularizer,
                                              name='layer2_MLP')
            self.layer2_MLP = tf.layers.dropout(self.layer2_MLP, rate=self.dropout)
            # 就是你在训练的时候想拿掉多少神经元，按比例计算。0就是没有dropout，1就是整个层都没了

            self.layer3_MLP = tf.layers.dense(inputs=self.layer2_MLP,
                                              units=self.embed_size // 4,
                                              activation=self.activation_func,
                                              kernel_initializer=self.initializer,
                                              kernel_regularizer=self.regularizer,
                                              name='layer3_MLP')
            self.layer3_MLP = tf.layers.dropout(self.layer3_MLP, rate=self.dropout)
            self.logits = tf.layers.dense(inputs= self.layer3_MLP,
                                          units = 1,
                                          activation=None,
                                          kernel_initializer=self.initializer,
                                          kernel_regularizer=self.regularizer,
                                          name='predict')
            self.logits_dense = tf.reshape(self.logits,[-1])

    def add_loss(self):
        losses =tf.square(self.rate-self.logits_dense)
        self.loss = tf.reduce_sum(losses)

    def add_train_step(self):
        '''
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr = tf.train.exponential_decay(self.lr, global_step,
                                             self.decay_steps, self.decay_rate, staircase=True)
        '''
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_step = optimizer.minimize(self.loss)

    def init_sess(self):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True
        self.sess = tf.Session(config=self.config)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        if os.path.exists(self.checkPoint):
            [os.remove(f) for f in os.listdir(self.checkPoint)]
        else:
            os.mkdir(self.checkPoint)

    def run(self):
        best_hr = -1
        best_NDCG = -1
        best_epoch = -1
        loss = np.inf
        print("Start Training!")
        for epoch in range(self.maxEpochs):
#            print("="*20+"Epoch ", epoch, "="*20)
            loss_temp=self.run_epoch(self.sess)
            if loss_temp < loss:
                loss = loss_temp
            else:break
            print("Epoch:",epoch,"loss:",loss_temp)
            self.saver.save(self.sess, self.checkPoint+'model.ckpt')
        print("Training complete!")

    def run_epoch(self, sess, verbose=10):
        train_u, train_i, train_r = self.dataSet.getInstances(self.train)

        # train_set = {'user':train_u,"item":train_i,'rate':train_r}
        # dataset = tf.data.Dataset.from_tensor_slices(train_set)
        # dataset = dataset.shuffle(100000).batch(self.batchSize)
        # iterator = tf.data.Iterator.from_structure(dataset.output_types,
        #                                            dataset.output_shapes)
        # sess.run(iterator.make_initializer(dataset))
        train_len = len(train_u)
        shuffled_idx = np.random.permutation(np.arange(train_len))
        train_u = train_u[shuffled_idx]
        train_i = train_i[shuffled_idx]
        train_r = train_r[shuffled_idx]

        num_batches = len(train_u) // self.batchSize + 1

        losses = []
        for i in range(num_batches):
            min_idx = i * self.batchSize
            max_idx = np.min([train_len, (i+1)*self.batchSize])
            train_u_batch =np.array(train_u[min_idx: max_idx])
            train_i_batch = train_i[min_idx: max_idx]
            train_r_batch = train_r[min_idx: max_idx]
            print("ssdsdsdds",train_u_batch.shape)
            print(train_u_batch)



            feed_dict = self.create_feed_dict(train_u_batch, train_i_batch, train_r_batch,self.drop)
            _, tmp_loss = sess.run([self.train_step, self.loss], feed_dict=feed_dict)
            losses.append(tmp_loss)
        loss = np.mean(losses)
        print("\nMean loss in this epoch is: {}".format(loss))
        return loss

    def create_feed_dict(self, u, i, r=None, drop=None):
        return {self.user: u,
                self.item: i,
                self.rate: r,
                self.drop: drop}

    def evaluate(self, sess, topK):
        def getHitRatio(ranklist, targetItem):
            for item in ranklist:
                if item == targetItem:
                    return 1
            return 0
        def getNDCG(ranklist, targetItem):
            for i in range(len(ranklist)):
                item = ranklist[i]
                if item == targetItem:
                    return math.log(2) / math.log(i+2)
            return 0


        hr =[]
        NDCG = []
        testUser = self.testNeg[0]
        testItem = self.testNeg[1]
        for i in range(len(testUser)):
            target = testItem[i][0]
            feed_dict = self.create_feed_dict(testUser[i], testItem[i])
            predict = sess.run(self.y_, feed_dict=feed_dict)

            item_score_dict = {}

            for j in range(len(testItem[i])):
                item = testItem[i][j]
                item_score_dict[item] = predict[j]

            ranklist = heapq.nlargest(topK, item_score_dict, key=item_score_dict.get)

            tmp_hr = getHitRatio(ranklist, target)
            tmp_NDCG = getNDCG(ranklist, target)
            hr.append(tmp_hr)
            NDCG.append(tmp_NDCG)
        return np.mean(hr), np.mean(NDCG)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('-dataName', action='store', dest='dataName', default='ml-1m')
    parser.add_argument('-negNum', action='store', dest='negNum', default=7, type=int)
    # parser.add_argument('-reg', action='store', dest='reg', default=1e-3)
    parser.add_argument('-lr', action='store', dest='lr', default=0.001)
    parser.add_argument('-maxEpochs', action='store', dest='maxEpochs', default=5, type=int)
    parser.add_argument('-batchSize', action='store', dest='batchSize', default=256, type=int)
    parser.add_argument('-earlyStop', action='store', dest='earlyStop', default=5)
    parser.add_argument('-checkPoint', action='store', dest='checkPoint', default='./checkPoint/')
    parser.add_argument('-topK', action='store', dest='topK', default=10)
    parser.add_argument('-optim',action='store',dest='optim',default='Adam')
    parser.add_argument('-initializer',action='store',dest='initializer',default="Xavier")
    parser.add_argument('-regularizer',action='store',dest='regularizer',default=0.0)
    parser.add_argument('-dropout',action='store',dest='dropout',default=0.0)
    parser.add_argument('-activation',action='store',dest='activation',default='ReLU')
    parser.add_argument('-embed_size',action='store',dest='embed_size',default=64,type=float)

    args = parser.parse_args()

    classifier = Model(args)

    classifier.run()
