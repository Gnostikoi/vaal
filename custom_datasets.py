from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy

# from ..HAABSA.utils import load_inputs_twitter, load_w2v
# from ..HAABSA.LcrModelAlt import lcr_rot
# from ..HAABSA.config import *
from utils.load_data import LoadSE5
import tensorflow as tf

def imagenet_transformer():
    transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def cifar10_transformer():
    return torchvision.transforms.Compose([
           torchvision.transforms.RandomHorizontalFlip(),
           torchvision.transforms.ToTensor(),
           transforms.Normalize(mean=[0.5, 0.5, 0.5,],
                                std=[0.5, 0.5, 0.5]),
       ])

class SemEvalRes(Dataset):
    def __init__(self, type, mlb=None):
        se5 = LoadSE5('restaurant', 1, False, 'USE_multilingual', mlb=mlb)
        if type == 'train':
            self.data, _ = se5.load_X(train=True, test=False)
            self.target, _ = se5.load_y(train=True, test=False)
        else:
            _, self.data = se5.load_X(train=False, test=True)
            _, self.target = se5.load_y(train=False, test=True)

        self.target = self.target.astype(numpy.float32)
        self.emb_size = self.data.shape[1]
        self.n_classes = self.target.shape[1]
        self.mlb = se5.mlb
                
    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)

        return self.data[index], self.target[index], index

    def __len__(self):
        return len(self.data)


class SemEvalRes5(Dataset):
    def __init__(self, path, sess):
        with tf.device('/gpu:1'):
            word_id_mapping, w2v = load_w2v(FLAGS.embedding_path, FLAGS.embedding_dim)
            word_embedding = tf.constant(w2v, name='word_embedding')
            keep_prob1 = tf.constant(FLAGS.keep_prob1, tf.float32)
            keep_prob2 = tf.constant(FLAGS.keep_prob2,tf.float32)
            l2 = FLAGS.l2_reg
            lambda_0 = tf.constant(FLAGS.lambda_0, tf.float32)  

            with tf.name_scope('inputs'):
                y_sen = tf.placeholder(tf.float32, [None, FLAGS.n_class], name='y_sentence_level')
                n_asp = tf.placeholder(tf.int32, [None], name='n_asp')

                x = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len], name='x')
                y = tf.placeholder(tf.float32, [None, FLAGS.n_class], name='y')
                sen_len = tf.placeholder(tf.int32, None, name='sentence_length')

                x_bw = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len], name='x_backwards')
                sen_len_bw = tf.placeholder(tf.int32, [None], name='sentence_length_backwards')

                target_words = tf.placeholder(tf.int32, [None, FLAGS.max_target_len], name='target_words')
                tar_len = tf.placeholder(tf.int32, [None], name='target_length')


            inputs_fw = tf.nn.embedding_lookup(word_embedding, x)
            inputs_bw = tf.nn.embedding_lookup(word_embedding, x_bw)
            target = tf.nn.embedding_lookup(word_embedding, target_words)

        x_f, sen_len, x_bw, sen_len_bw, yi, y_sen, target, tl, _, _, _ , _n_asp= load_inputs_twitter(
            train_path,
            word_id_mapping,
            FLAGS.max_sentence_len,
            'TC',
            is_r, # reverse
            FLAGS.max_target_len
        )

        prob, prob_sen, output, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r = lcr_rot(n_asp, inputs_fw, inputs_bw, sen_len, sen_len_bw, target, tl, keep_prob1, keep_prob2, l2, 'all')
        with sess.as_default():
            outputs = sess.run(output, feed_dict = {
                        x: x_f[r_index],
                        x_bw: x_b[r_index],
                        y: yi[r_index],
                        y_sen: y_sen_i[index],
                        n_asp: _n_asp,
                        sen_len: sen_len_f[r_index],
                        sen_len_bw: sen_len_b[r_index],
                        target_words: target[r_index],
                        tar_len: tl[r_index]
                    })
        self.n_asp = _n_asp
        self.data = outputs
        self.target_sen = y_sen
        self.target = yi

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)

        r_index = self.n_asp[index]
        data, target, target_sen = self.data[r_index], self.target[r_index], self.target_sen[index]

        return data, target, target_sen, index

    def __len__(self):
        return len(self.target_sen)


class CIFAR10(Dataset):
    def __init__(self, path):
        self.cifar10 = datasets.CIFAR10(root=path,
                                        download=True,
                                        train=True,
                                        transform=cifar10_transformer())

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)

        data, target = self.data[index], self.target[index]

        return data, target, index

    def __len__(self):
        return len(self.cifar10)


class CIFAR100(Dataset):
    def __init__(self, path):
        self.cifar100 = datasets.CIFAR100(root=path,
                                        download=True,
                                        train=True,
                                        transform=cifar10_transformer())

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)

        data, target = self.cifar100[index]

        # Your transformations here (or set it in CIFAR10)

        return data, target, index

    def __len__(self):
        return len(self.cifar100)


class ImageNet(Dataset):
    def __init__(self, path):
        self.imagenet = datasets.ImageFolder(root=path, transform=imagenet_transformer)

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)
        data, target = self.imagenet[index]

        return data, target, index

    def __len__(self):
        return len(self.imagenet)
