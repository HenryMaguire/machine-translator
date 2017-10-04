import time
import pickle
import numpy as np
import tensorflow as tf
from utils import *
import os

word2id_s = load_obj("DATA/preprocess/word2id_s")
word2id_t = load_obj("DATA/preprocess/word2id_t")
id2word_s = load_obj("DATA/preprocess/id2word_s")
id2word_t = load_obj("DATA/preprocess/id2word_t")

text_source = load_obj("DATA/preprocess/source_train")[0:10000] # Preprocessed already
text_target = load_obj("DATA/preprocess/target_train")[0:10000] # Preprocessed already

vocab_size_t = len(word2id_t.keys())
vocab_size_s = len(word2id_s.keys())
lang_s = 'fr'
lang_t = 'en'


def get_embeddings(id_to_word, lang):
    # We load pretrained word2vec embeddings from polyglot to save on training time
    filename ='DATA/polyglot-'+lang+'.pkl'
    pretrain_vocab, pretrain_embed = pickle.load(open(filename, 'rb'))
    embed_vocab = [pretrain_embed[pretrain_vocab.index('<PAD>')], pretrain_embed[pretrain_vocab.index('</S>')]]
    skip_count = 0
    skipped_words = []
    for idx, word in sorted(id_to_word.items()[2::]):
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
    return embed_vocab, skipped_words
# the ith word in words corresponds to the ith embedding

embed_vocab_s, skipped_s = get_embeddings(id2word_s, lang=lang_s)
embed_vocab_t, skipped_t = get_embeddings(id2word_t, lang=lang_t)

#from utils import format_batch
test_x = [[5,2,3],[2], [4,2], [1,2]]
# it's going to go from the number of cols being the sequence length/ num of rows being batch size
# to the number of rows being the max sequence length/ num cols being batch size
# Essentially like a padding and then transpose
tf.reset_default_graph()
input_embedding_size = 64 # Fixed due to pretrained embedding files
encoder_hidden_units = 256

decoder_hidden_units = encoder_hidden_units # Must be the same at the moment
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
encoder_inputs_embedded = tf.nn.embedding_lookup(embed_vocab_s, encoder_inputs)
decoder_inputs_embedded = tf.nn.embedding_lookup(embed_vocab_t, decoder_inputs)

encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)

encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded,
                                                         dtype=tf.float32, time_major=True)
decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)

decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
                                decoder_cell, decoder_inputs_embedded,
                                initial_state=encoder_final_state,
                                dtype=tf.float32, time_major=True,
                                scope="plain_decoder")
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

timestep_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size_t, dtype=tf.float32),
    logits=decoder_logits,
)
print timestep_cross_entropy
# loss is the mean of the cross entropy
loss = tf.reduce_mean(timestep_cross_entropy)
print loss
# We use AdaM which combines AdaGrad (parameters updated less often get updated more strongly)
# and momentum (updates depend on the slope of previous updates - avoiding local minima)
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



def make_test_feed_dict(fd_keys,s_batch, t_batch, reverse_encoder_inputs= False):
    # At testing time, we can't supervise the decoder layer with
    # the 'gold truth' example as input, so we instead feed in
    # word generated at  previous timestep. This is (apparently)
    # equivalent to feeding in zeros for the decoder inputs
    encoder_inputs_ = format_batch(s_batch)
    if reverse_encoder_inputs:
        encoder_inputs_ = format_batch([sequence[-2::-1]+[1] for sequence in s_batch])
    decoder_targets_ = format_batch([sequence for sequence in t_batch])
    decoder_inputs_ = format_batch([[0]*len(sequence) for sequence in t_batch])
    return {
        fd_keys[0]: encoder_inputs_,
        fd_keys[1]: decoder_inputs_,
        fd_keys[2]: decoder_targets_,
    }


NUMCORES=int(os.environ["NSLOTS"])
save_model_path = './rnn_nmt'
epochs = 1 # How many times we loop over the whole training data
batch_size = 92 # After how many sequences do we update the weights?
print "there will be {} samples in the final batch".format(len(source_train)%batch_size)
fd_keys = [encoder_inputs, decoder_inputs, decoder_targets]
loss_list = []
print('Training...')
with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUMCORES,
                            intra_op_parallelism_threads=NUMCORES)) as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    try:
        batch_n = 0
        ti = time.time()
        print "training has begun..."
        for epoch in range(epochs):
            ti = time.time()
            for s_batch, t_batch in batch_source_target(source_train, target_train, batch_size):
                feed_dict = make_feed_dict(fd_keys, s_batch, t_batch)
                _, l = sess.run([train_op, loss], feed_dict)

                #if batch_n == 0 or batch_n == 60:
                #
                if (batch_n==0) or (batch_n%500) == 0:
                    loss_list.append(l)
                    print "epoch {}".format(epoch+1)
                    print 'batch {}'.format(batch_n)
                    print 'loss: {}'.format(sess.run(loss, feed_dict))
                    predict_ = sess.run(decoder_prediction, feed_dict)
                    i =1
                    for (inp, act, pred) in zip(feed_dict[encoder_inputs].T,
                                                             feed_dict[decoder_targets].T,
                                                             predict_.T)[11:12]:
                        print '  sample {}:'.format(i)
                        print '    input     : {} \n {}'.format(
                            format_idx(inp), ids_to_phrases(inp, id2word_s))
                        print '    actual     : {} \n {}'.format(
                            format_idx(act), ids_to_phrases(act, id2word_t))
                        print '    predicted     : {} \n {}'.format(
                            format_idx(pred), ids_to_phrases(pred, id2word_t))
                        i+=1
                batch_n += 1
            print "Epoch took {} seconds to complete.".format(time.time()-ti)
            saver = tf.train.Saver()
            save_path = saver.save(sess, save_model_path)
        print 'Training is complete'
    except KeyboardInterrupt:
        print 'training interrupted'
save_obj(loss_track)
