import tensorflow as tf
from data_helpers import loadDataset, getBatches, sentence2enco
from model import Seq2SeqModel
from tqdm import tqdm
import math
import os

# Define optional parameters for command line ################################################################
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

# Get parameters out for usage in this file 
FLAGS = tf.app.flags.FLAGS

# Load data(word-id pairs, id-word pairs, question-answer pairs) from the data path ############################
data_path = r'D:\Northeastern Semester 1\NLP\seq2seq\data\dataset-cornell-length10-filter1-vocabSize40000.pkl'
word2id, id2word, trainingSamples = loadDataset(data_path)

# Use session to manage the whole train process, see below to get rough ideas ##################################
# https://blog.csdn.net/weixin_40208575/article/details/100047481
# https://zhuanlan.zhihu.com/p/87299728
# https://blog.csdn.net/SA14023053/article/details/51890076
# Feel free to translate if you are an English learner
with tf.Session() as sess:

    # Define a Seq2Seq model -----------------------------------------------------------------------------------
    model = Seq2SeqModel(FLAGS.rnn_size, FLAGS.num_layers, FLAGS.embedding_size, FLAGS.learning_rate, word2id,
                         mode='train', use_attention=True, beam_search=False, beam_size=5, max_gradient_norm=5.0)

    # GET the saved model according to the checkpoint file under model_dir path --------------------------------
    # ckpt contains two attributes, model_checkpoint_path, all_model_checkpoint path
    # model_checkpoint_path means the latest model
    # IF all attributes exist, we restore parameters from the model file, 
    # OTHEWISE, initialize the new global variables 
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('Created new model parameters..')
        sess.run(tf.global_variables_initializer())
    
    # Define helper variables ----------------------------------------------------------------------------------
    # Define current step for saving models or printing loss every steps_per_checkpoint
    current_step = 0
    
    # Record summary for visualization in tensorboard
    summary_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

    # Start the train process ----------------------------------------------------------------------------------
    # Iterate epoches
    for e in range(FLAGS.numEpochs):  
        # Print the current epoch starting from Epoch 1
        print("----- Epoch {}/{} -----".format(e + 1, FLAGS.numEpochs))  
        # Get padded question-answer batches
        batches = getBatches(trainingSamples, FLAGS.batch_size)

        # Iterate batches
        for nextBatch in tqdm(batches, desc="Training"):  # processing bar, tqdm
            # Train the model using each batch
            loss, summary = model.train(sess, nextBatch)
            # Current step starting from 1
            current_step += 1
            # Every steps_percheckpoint
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Write current step, loss, perplexity into tqdm, and summary_writer for visulization
                perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                tqdm.write("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (current_step, loss, perplexity))
                summary_writer.add_summary(summary, current_step)
                # Save the model state suffixed with current_step into checkpoint_path
                checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
                model.saver.save(sess, checkpoint_path, global_step=current_step)