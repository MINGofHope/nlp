import tensorflow as tf
from data_helpers import loadDataset, getBatches, sentence2enco
from model import Seq2SeqModel
import sys
import numpy as np


tf.app.flags.DEFINE_integer('rnn_size', 256, 'Number of hidden units in each layer')  # 1024
tf.app.flags.DEFINE_integer('num_layers', 3, 'Number of layers in each encoder and decoder')  # 2
tf.app.flags.DEFINE_integer('embedding_size', 256, 'Embedding dimensions of encoder and decoder inputs')  # 1024

tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')  # 0.0001
tf.app.flags.DEFINE_integer('batch_size', 64, 'Batch size')  # 128  
tf.app.flags.DEFINE_integer('numEpochs', 15, 'Maximum # of training epochs')  # 30
tf.app.flags.DEFINE_integer('steps_per_checkpoint', 100, 'Save model checkpoint every this iteration')  # 100
tf.app.flags.DEFINE_string('model_dir', 'model/', 'Path to save model checkpoints')
tf.app.flags.DEFINE_string('model_name', 'chatbot.ckpt', 'File name used for model checkpoints')
FLAGS = tf.app.flags.FLAGS

data_path = r'D:\Northeastern Semester 1\NLP\seq2seq\data\dataset-cornell-length10-filter1-vocabSize40000.pkl'
word2id, id2word, trainingSamples = loadDataset(data_path)

def predict_ids_to_seq(predict_ids, id2word, beam_szie):
    '''
    将beam_search返回的结果转化为字符串
    :param predict_ids: 列表，长度为batch_size，每个元素都是decode_len*beam_size的数组
    :param id2word: vocab字典
    :return:
    '''
    for single_predict in predict_ids:
        for i in range(beam_szie):
            predict_list = np.ndarray.tolist(single_predict[:, :, i])
            predict_seq = [id2word[idx] for idx in predict_list[0]]
            print(" ".join(predict_seq))

with tf.Session() as sess:
    model = Seq2SeqModel(FLAGS.rnn_size, FLAGS.num_layers, FLAGS.embedding_size, FLAGS.learning_rate, word2id,
                         mode='decode', use_attention=True, beam_search=False, beam_size=5, max_gradient_norm=5.0)
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise ValueError('No such file:[{}]'.format(FLAGS.model_dir))
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
        batch = sentence2enco(sentence, word2id)
        predicted_ids = model.infer(sess, batch)
        predict_ids_to_seq(predicted_ids, id2word, 1)
        # out_str = ''.join([id2word[idx] for idx in predicted_ids[0]])
        print("> ", "")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
