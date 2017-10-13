"""
floyd run --gpu --env tensorflow-1.1:py2 --data hblorm/datasets/nmt_data/1:/DATA "python machine-translator-gpu.py"
"""
import time
import pickle
import numpy as np
import codecs
from utils import *
import os
import logging
import sys
reload(sys)
sys.setdefaultencoding('utf8')
#logging.getLogger("tensorflow").setLevel(logging.ERROR)

logging.basicConfig(level=logging.DEBUG, filename='logfile', filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
encoder_hidden_units = 512
reverse_encoder_inputs = True
test_only = True

log_dir = "trained/nmt_b512"
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def truncate_dict(previous, num_words=1000):
    d = dict()
    for i in previous.keys()[0:num_words]:
        d[i] = previous[i]
    return d


dev_length = 75000
data_path = "DATA/"
data_dict = load_obj(data_path+'data_dic')
s_train = data_dict["s_train"][0::]
t_train = data_dict["t_train"][0::]
s_test = data_dict["s_test"]
t_test = data_dict["t_test"]
word2id_s = data_dict["word2id_s"]
word2id_t = data_dict["word2id_t"]
id2word_s = data_dict["id2word_s"]
id2word_t = data_dict["id2word_t"]
train_length = len(s_train)
vocab_size_t = len(word2id_t.keys())
vocab_size_s = len(word2id_s.keys())
lang_s = 'fr'
lang_t = 'en'

logging.info("Preprocessed data loaded...")
print("Preprocessed data loaded...")
def get_embeddings(id_to_word, lang):
    # We load pretrained word2vec embeddings from polyglot to save on training time
    filename =data_path+'polyglot-'+lang+'.pkl'
    pretrain_vocab, pretrain_embed = pickle.load(open(filename, 'rb'))
    embed_vocab = [pretrain_embed[pretrain_vocab.index('<PAD>')], pretrain_embed[pretrain_vocab.index('</S>')]]
    skip_count = 0
    skipped_words = []
    metadata = u'<PAD>\n<EOS>\n'
    for idx, word in sorted(id_to_word.items()[2::]):
        metadata += word+r"\n"
        try:
            pretrain_idx = pretrain_vocab.index(word)
            embed_vocab.append(pretrain_embed[pretrain_idx])
        except ValueError:
            try:
                # it could be that the word is a name which needs to
                # be capitalized. Try this...
                pretrain_idx = pretrain_vocab.index(str(word.title()))
                embed_vocab.append(pretrain_embed[pretrain_idx])
            except ValueError:
                try:
                    # it could be that the word is an achronym which needs to
                    # be upper case. Try this...
                    pretrain_idx = pretrain_vocab.index(word.upper())
                    embed_vocab.append(pretrain_embed[pretrain_idx])
                except ValueError:
                    # Give up trying to find an embedding.
                    # How many words are skipped? Which ones?
                    skip_count +=1
                    skipped_words.append(word)
                    # Let's just initialise the embedding to a random normal distribution
                    embed_vocab.append(np.random.normal(loc=0.0, scale=np.sqrt(2)/4, size=64))

    embed_vocab = np.array(embed_vocab, dtype=np.float32)
    print "The embedding matrix for {} has {} columns and {} rows.".format(lang,
                                                embed_vocab.shape[0], embed_vocab.shape[1])
    print "{} vocab words were not in the {} embeddings file.".format(skip_count, lang)
    return embed_vocab, skipped_words, metadata
# the ith word in words corresponds to the ith embedding
embed_vocab_s = np.array(np.random.normal(loc=0.0, scale=np.sqrt(2)/4, size=(vocab_size_s, 64)), dtype=np.float32)
embed_vocab_t = np.array(np.random.normal(loc=0.0, scale=np.sqrt(2)/4, size=(vocab_size_t, 64)), dtype=np.float32)
print embed_vocab_t
if not test_only:
    embed_vocab_s, skipped_s, metadata_s = get_embeddings(id2word_s, lang=lang_s)
    embed_vocab_t, skipped_t, metadata_t = get_embeddings(id2word_t, lang=lang_t)

#from utils import format_batch
#test_x = [[5,2,3],[2], [4,2], [1,2]]
# it's going to go from the number of cols being the sequence length/ num of rows being batch size
# to the number of rows being the max sequence length/ num cols being batch size
# Essentially like a padding and then transpose
def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
  I'll just get as much data as I can."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

tf.reset_default_graph()
input_embedding_size = 64 # Fixed due to pretrained embedding files

decoder_hidden_units = encoder_hidden_units # Must be the same at the moment
with tf.device("/cpu:0"):
    with tf.name_scope("source_embeddings"):
        source_embeddings = tf.Variable(embed_vocab_s, trainable=True, name="source_embeddings")
        variable_summaries(source_embeddings)
    with tf.name_scope("target_embeddings"):
        target_embeddings = tf.Variable(embed_vocab_t, trainable=True, name="target_embeddings")
        variable_summaries(target_embeddings)
    with tf.name_scope('encoder_inputs'):
        encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
    with tf.name_scope('decoder_targets'):
        decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
    with tf.name_scope('decoder_inputs'):
        decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
    with tf.name_scope('encoder_inputs_embed'):
        encoder_inputs_embedded = tf.nn.embedding_lookup(source_embeddings,
                                                encoder_inputs, name='encoder_inputs_embed')

    with tf.name_scope('decoder_inputs_embed'):
        decoder_inputs_embedded = tf.nn.embedding_lookup(target_embeddings,
                                                decoder_inputs, name='decoder_inputs_embed')
    with tf.name_scope('encoder_cell'):
        encoder_cell = tf.contrib.rnn.GRUCell(encoder_hidden_units)

        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                            encoder_cell, encoder_inputs_embedded,
                            dtype=tf.float32, time_major=True)
        variable_summaries(encoder_outputs)
        variable_summaries(encoder_final_state)

    with tf.name_scope('decoder_cell'):
        decoder_cell = tf.contrib.rnn.GRUCell(decoder_hidden_units)
        decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
    		                decoder_cell, decoder_inputs_embedded,
    		                initial_state=encoder_final_state,
    		                dtype=tf.float32, time_major=True,
    		                scope="plain_decoder")
        variable_summaries(decoder_outputs)
        variable_summaries(decoder_final_state)
    with tf.name_scope('projection_layer'):
        decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size_t)
    """
    #weights
    W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size_t], -0.5, 0.5), dtype=tf.float32)
    #bias
    b = tf.Variable(tf.zeros([vocab_size_t]), dtype=tf.float32)
    decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
    # why do we only flatten the tensor so it's rank 2?
    decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
    #feed flattened tensor through projection
    decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
    # make the logits the shape of the
    decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size_t))
	"""
    with tf.name_scope('prediction'):
        decoder_prediction = tf.argmax(decoder_logits, axis=2)
    with tf.name_scope('timestep_cross_entropy'):
        timestep_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                                            labels=tf.one_hot(decoder_targets,
                                            depth=vocab_size_t, dtype=tf.float32),
                                            logits=decoder_logits)
        variable_summaries(timestep_cross_entropy)
    print timestep_cross_entropy
    # loss is the mean of the cross entropy
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(timestep_cross_entropy)
        variable_summaries(loss)
        tf.summary.scalar('loss', loss)
    # We use AdaM which combines AdaGrad (parameters updated less often get updated more 	strongly)
    # and momentum (updates depend on the slope of previous updates - avoiding local minima)
    with tf.name_scope('Optimizer'):
        train_op = tf.train.AdamOptimizer().minimize(loss)

def batch_source_target(source, target, batch_size):
    assert len(source) == len(target)
    for start in range(0, len(source), batch_size):
        end = min(start + batch_size, len(source))
        #print type(source[start:end])
        #print len(target[start:end])
        yield source[start:end], target[start:end]


def make_feed_dict(fd_keys, s_batch, t_batch, reverse_encoder_inputs= False):
    encoder_inputs_ = format_batch(s_batch)
    if reverse_encoder_inputs:
        encoder_inputs_ = format_batch([sequence[-2::-1]+[1] for sequence in s_batch])
    decoder_inputs_ = format_batch([[1]+sequence[0:-1] for sequence in t_batch])
    decoder_targets_ = format_batch([sequence for sequence in t_batch])
    return {
        fd_keys[0]: encoder_inputs_,
        fd_keys[1]: decoder_inputs_,
        fd_keys[2]: decoder_targets_,
    }



def make_test_feed_dict(fd_keys,s_batch,t_batch, reverse_encoder_inputs= False):
    # At testing time, we can't supervise the decoder layer with
    # the 'gold truth' example as input, so we instead feed in
    # word generated at  previous timestep. This is (apparently)
    # equivalent to feeding in zeros for the decoder inputs
    encoder_inputs_ = format_batch(s_batch)
    if reverse_encoder_inputs:
        encoder_inputs_ = format_batch([sequence[-2::-1]+[1] for sequence in s_batch])
    decoder_targets_ = format_batch([sequence for sequence in t_batch])
    decoder_inputs_ = format_batch([[1]*len(sequence) for sequence in t_batch])
    return {
        fd_keys[0]: encoder_inputs_,
        fd_keys[1]: decoder_inputs_,
        fd_keys[2]: decoder_targets_,
    }

#NUM_CORES = 3
epochs = 120 # How many times we loop over the whole training data
batch_size = 92 # After how many sequences do we update the weights?

merged = tf.summary.merge_all()
total_batches = train_length/batch_size
samples_in_final = train_length%batch_size
print "there will be {} batches and {} samples in the final batch".format(total_batches, samples_in_final)
fd_keys = [encoder_inputs, decoder_inputs, decoder_targets]
loss_list = []
print('Training...')
#with tf.Session(config=tf.ConfigProto(
#  intra_op_parallelism_threads=NUM_CORES)) as sess:
"""
def save_metadata(metadata, filename):
    with open(filename, 'w') as f:
        f.write(metadata.encode('utf-8'))

with tf.Session(config=tf.ConfigProto(log_device_placement=True,
                                    allow_soft_placement=True)) as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(log_dir+'/train',
                                        sess.graph)
    summary_writer = tf.summary.FileWriter(log_dir,
                                        sess.graph)
    config = projector.ProjectorConfig()
    embedding_enc = config.embeddings.add()
    embedding_enc.tensor_name = source_embeddings.name
    # Link this tensor to its metadata file (e.g. labels).
    meta_s_filename = os.path.join(log_dir, 'metadata_s.tsv')
    save_metadata(metadata_s, meta_s_filename)
    embedding_enc.metadata_path = meta_s_filename

    embedding_dec = config.embeddings.add()
    embedding_dec.tensor_name = target_embeddings.name
    # Link this tensor to its metadata file (e.g. labels).
    meta_t_filename = os.path.join(log_dir, 'metadata_t.tsv')
    save_metadata(metadata_t, meta_t_filename)
    embedding_dec.metadata_path = meta_t_filename

    # Saves a configuration file that TensorBoard will read during startup.
    projector.visualize_embeddings(summary_writer, config)
    # Training cycle
    try:
        batch_n = 0
        ti = time.time()
        print "training has begun..."
        for epoch in range(1, epochs+1):
            ti = time.time()
            saver = tf.train.Saver()
            if epoch%10==0:
                save_path = saver.save(sess, log_dir+"/model.ckpt", epoch)
            for s_batch, t_batch in batch_source_target(s_train,
                                                        t_train, batch_size):
                feed_dict = make_feed_dict(fd_keys,s_batch, t_batch,
                                reverse_encoder_inputs=reverse_encoder_inputs)
                _, l = sess.run([train_op, loss], feed_dict)
                if (batch_n==0) or (batch_n%270) == 0:
                    loss_list.append(l)
                    logging.info("epoch {}".format(epoch))
                    logging.info('batch {}'.format(batch_n-(epoch-1)*total_batches))
                    logging.info('loss: {}'.format(sess.run(loss, feed_dict)))
                    print "**********************************************"*5
                    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"*5
                    print ("epoch {}".format(epoch))
                    print ('batch {}'.format(batch_n-(epoch-1)*total_batches))
                    print ('loss: {}'.format(sess.run(loss, feed_dict)))
                    summary_, predict_ = sess.run([merged, decoder_prediction], feed_dict)
                    summary_writer.add_summary(summary_, batch_n)
                    i = 1
                    rand_sample = np.random.randint(0,batch_size-2)
                    for (inp, act, pred) in zip(feed_dict[encoder_inputs].T,
                                                 feed_dict[decoder_targets].T,
                                                 predict_.T)[rand_sample:rand_sample+2]:
                        logging.info('  sample {}:'.format())
                        logging.info('    input     : {} \n {}'.format(
                                         format_idx(inp), ids_to_phrases(inp, id2word_s)))
                        logging.info('    actual     : {} \n {}'.format(
                                         format_idx(act), ids_to_phrases(act, id2word_t)))
                        logging.info('    predicted     : {} \n {}'.format(
                                         format_idx(pred), ids_to_phrases(pred, id2word_t)))
                        print ('  sample {}:'.format(i))
                        print ('    input     : {} \n {}'.format(
                                         format_idx(inp), ids_to_phrases(inp, id2word_s)))
                        print ('    actual     : {} \n {}'.format(
                                         format_idx(act), ids_to_phrases(act, id2word_t)))
                        print ('    predicted     : {} \n {}'.format(
                                         format_idx(pred), ids_to_phrases(pred, id2word_t)))
                        i+=1
                batch_n += 1
            logging.info("Epoch {} took {} seconds to complete.".format(epoch, time.time()-ti))
            print ("Epoch {} took {} seconds to complete.".format(epoch, time.time()-ti))

            save_obj(loss_list, log_dir+"/loss_track")
        logging.info('Training is complete')
    except KeyboardInterrupt:
        print 'training interrupted'
"""
saver = tf.train.Saver()
from nltk.translate import bleu_score
print
with tf.Session() as sess:
    saver.restore(sess, log_dir+"/model.ckpt-20")
    predictions, actuals = [], []
    fd_keys = [encoder_inputs, decoder_inputs, decoder_targets]
    try:
        batch_n = 0
        print "testing has begun..."
        for s_batch, t_batch in batch_source_target(s_test, t_test, batch_size):
            feed_dict = make_test_feed_dict(fd_keys, s_batch, t_batch, reverse_encoder_inputs= True)
            predict_ = sess.run(decoder_prediction, feed_dict)
            for i, (inp, act, pred) in enumerate(zip(feed_dict[encoder_inputs].T,
                                                     feed_dict[decoder_targets].T,
                                                     predict_.T)):
                actuals.append([remove_EOS_PAD(act)])
                predictions.append(remove_EOS_PAD(pred))
                #print ('  sample {}:'.format(rand_sample))
                #print ('    input     : {} \n {}'.format(
                #                 format_idx(inp), ids_to_phrases(inp, id2word_s)))
                print ('    actual     : {} \n \t {}'.format(
                                 format_idx(act), ids_to_phrases(act, id2word_t)))
                print ('    predicted     : {} \n \t {}'.format(
                                 format_idx(pred), ids_to_phrases(pred, id2word_t)))

            batch_n+=1
        BLEU4 = bleu_score.corpus_bleu(actuals, predictions, weights=(0.25, 0.25, 0.25, 0.25))
        BLEU2 = bleu_score.corpus_bleu(actuals, predictions, weights=(0.5, 0.5))
        BLEU1 = bleu_score.corpus_bleu(actuals, predictions, weights=([1]))
        print'Testing is complete\nNMT Corpus BLEU4: {} \t BLEU2: {} \t BLEU1: {}\n'.format(BLEU4, BLEU2, BLEU1)
        #print "Benchmark BLEU4 : {} \t BLEU2: {} \t BLEU1: {}\n".format(BM_BLEU4, BM_BLEU2, BM_BLEU1)
    except KeyboardInterrupt:
        print 'testing interrupted'

    #print ids_to_phrases(actuals[8]+[0],id2word_t)
    #print ids_to_phrases(predictions[8]+[0],id2word_t)
