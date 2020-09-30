import numpy as np
import pandas as pd
import tensorflow as tf
import os

DATA_DIR = 'data/ratings.dat'
DATA_PATH = 'data/'
COLUMN_NAMES = ['user','item','rate']

def re_index(s):
    i = 0
    s_map = {}
    for key in s:
        s_map[key] = i
        i += 1
    return s_map

def load_data():
    full_data = pd.read_csv(DATA_DIR,sep='::',header=None,names=COLUMN_NAMES,
                            usecols=[0,1,2],dtype={0:np.int32,1:np.int32,2:np.int32},engine='python')  #只取了user_ID,item_ID

# 3,900 movies ，6,040 MovieLens users
    full_data.user = full_data['user'] - 1              #对“user”这一列都减一
                                                    # full_data['user']和full_data.user是一样的
                                                    #full_data.user = full_data['user'] - 1   == full_data["user"] = full_data['user'] - 1

    user_set = set(full_data['user'].unique())         #user的集合 {0->6039}共9040
    item_set = set(full_data['item'].unique())         #item集合{1->3952} ,有缺的，里面只有3706个数
    #print("item_set1",item_set)

    user_size = len(user_set)   #用户数，项目数
    item_size = len(item_set)

    item_map = re_index(item_set)  #对item建立字典{item_ID:index}
    item_list = []

    full_data['item'] = full_data['item'].map(lambda x:item_map[x])#做一个映射，将有顺序缺损的item_id转为连续的数字，在替换掉full_data["item"]

    item_set = set(full_data.item.unique()) #{0->3705}
    #print("item_set2", item_set)

    user_bought = {}
    for i in range(len(full_data)):#统计每个用户有交互的item
        u = full_data['user'][i]
        t = full_data['item'][i]
        if u not in user_bought:
            user_bought[u] = []
        user_bought[u].append(t)
                               #user_bought：{User_ID:[有交互的item_ID,item_ID.........}
    #print("user1_len",len(user_bought[1]))
    user_negative = {}
    for key in user_bought:
        user_negative[key] = list(item_set - set(user_bought[key])) #使用所有的item减去有交互的item得到没有交互的item


    user_length = full_data.groupby('user').size().tolist()#分组聚合,统计每个用户的交互的item的数量
    #print("user_length:  ",user_length)
    split_train_test = []

    for i in range(len(user_set)):
        for _ in range(user_length[i] - 1): #取每个用户的最后一个item作为测试集，其他的都是训练集
                                            #例如user0,他有53个交互item, 这里的训练是从（0->52）包含52，剩下一个53作为测试集
            split_train_test.append('train')
        split_train_test.append('test')

    full_data['split'] = split_train_test
    """
    full_data变成了这样
     index   user   item  split
       0     0       0    train
       1     0       12   test
    """
    train_data = full_data[full_data['split'] == 'train'].reset_index(drop=True)#drop=True: 把原来的索引index列去掉
    test_data = full_data[full_data['split'] == 'test'].reset_index(drop=True)
    del train_data['split']
    del test_data['split']
    print("train_data:",train_data)
    print("test_data:",test_data)

    labels  = np.ones(len(train_data),dtype=np.int32) #训练集里面的标签都为1
    # labels = full_data['rate']

    train_features = train_data
    train_labels = labels.tolist()

    test_features = test_data
    test_labels = test_data['item'].tolist()

    return ((train_features, train_labels),
            (test_features, test_labels),
            (user_size, item_size),
            (user_bought, user_negative))

# NCF_input.train_input_fn(train_features,train_labels,FLAGS.batch_size,user_negative,FLAGS.negative_num)
#(features,labels,batch_size,user_negative,num_neg)
# dump_data(features, labels, user_negative, num_neg, True)
#add_negative(features,user_negative,labels,num_neg,is_training)
def add_negative(features,user_negative,labels,numbers,is_training):#features=train_features
    feature_user,feature_item,labels_add,feature_dict = [],[],[],{}

    for i in range(len(features)):
        user = features['user'][i]
        item = features['item'][i]
        label =features['rate'][i]

        feature_user.append(user)
        feature_item.append(item)
        labels_add.append(label)

        neg_samples = np.random.choice(user_negative[user],size=numbers,replace=False).tolist()

        if is_training:
            for k in neg_samples:
                feature_user.append(user)
                feature_item.append(k)
                labels_add.append(0)
             # """
             # feature_user[userid,userid]
             # feature_item[itemid,itemid]
             #       label[    1   0    ]
             #
             # """

        else: #向测试集里面加一些负样本
            for k in neg_samples:
                feature_user.append(user)
                feature_item.append(k)
                labels_add.append(k)


    feature_dict['user'] = feature_user
    feature_dict['item'] = feature_item

    return feature_dict,labels_add


# NCF_input.train_input_fn(train_features,train_labels,FLAGS.batch_size,user_negative,FLAGS.negative_num)
#(features,labels,batch_size,user_negative,num_neg)
# dump_data(features, labels, user_negative, num_neg, True)
def dump_data(features,labels,user_negative,num_neg,is_training):
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    """
    features:(train)
    user item
    0     1104
    0     23
    0     45
    0      1  
    features:(test)
    user  item
    0       25
    1       36
    2       56
    
    
    """
    features,labels = add_negative(features,user_negative,labels,num_neg,is_training)
    #features字典 {“user”:[user_Id,user_Id...............],"item":[item_id,item_id,item_id................]}

    data_dict = dict([('user', features['user']),
                      ('item', features['item']), ('label', labels)])

    print(data_dict)
    if is_training:
        np.save(os.path.join(DATA_PATH, 'train_data.npy'), data_dict)
        pd1= pd.DataFrame(data_dict)
        pd1.to_csv("data/train.csv",encoding="utf_8_sig")
    else:
        np.save(os.path.join(DATA_PATH, 'test_data.npy'), data_dict)
        pd1= pd.DataFrame(data_dict)
        pd1.to_csv("data/test.csv",encoding="utf_8_sig")


# NCF_input.train_input_fn(train_features,train_labels,FLAGS.batch_size,user_negative,FLAGS.negative_num)
def train_input_fn(features,labels,batch_size,user_negative,num_neg):
    data_path = os.path.join(DATA_PATH, 'train_data.npy')
    if not os.path.exists(data_path):
        dump_data(features, labels, user_negative, num_neg, True)

    data = np.load(data_path,allow_pickle=True).item()

    dataset = tf.data.Dataset.from_tensor_slices(data)
    """
    data字典 {“user”:[user_Id,user_Id...............],"item":[item_id,item_id,item_id................]。"label":[1,1,1,1,1,0,0,0.......]}
    from_tensor_slices让它变成了:
    {'user': 0, 'item': 1154, 'label': 1}
    {'user': 0, 'item': 3005, 'label': 1}
    {'user': 0, 'item': 2119, 'label': 1}
    {'user': 0, 'item': 1760, 'label': 1}
    {'user': 0, 'item': 1631, 'label': 1}
    {'user': 0, 'item': 260,  'label':  0}
    
    """



    dataset = dataset.shuffle(100000).batch(batch_size)
    print("dataset",dataset)
    """
    tf.data.Dataset.from_tensor_slices五步加载数据集:
    Step0: 准备要加载的numpy数据
    Step1: 使用 tf.data.Dataset.from_tensor_slices() 函数进行加载
    Step2: 使用 shuffle() 打乱数据
    Step3: 使用 map() 函数进行预处理
    Step4: 使用 batch() 函数设置 batch size 值
    Step5: 根据需要 使用 repeat() 设置是否循环迭代数据集
    
    shuffle(10000):10000是打乱的次数
    map():预处理，可以传入预处理的方法
    batch():数据批量大小 eg:128
    """
    return dataset


def eval_input_fn(features, labels, user_negative, test_neg):
    """ Construct testing dataset. """
    data_path = os.path.join(DATA_PATH, 'test_data.npy')
    if not os.path.exists(data_path):
        dump_data(features, labels, user_negative, test_neg, False)

    data = np.load(data_path,allow_pickle=True).item()
    print("Loading testing data finished!")
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.batch(test_neg + 1)
    ##我去
    #test没有打乱，一个批量就是一个user的测试集大小，所以label[0]就是与用户有过交集的那个item_id

    return dataset





