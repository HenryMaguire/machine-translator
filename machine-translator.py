import numpy as np
import pickle
from utils import *


def ids_to_phrases(idx_list, id_to_word):
    # Takes list of word ids and returns a string of words
    # Mainly for use in analysis
    phrase = ''
    id_dict = id_to_word
    i=0
    while idx_list[i] not in (1,0):
        phrase+= id_dict[idx_list[i]]+' '
        i+=1
    return phrase
# Test the functionality
print ids_to_phrases([234, 432, 102, 12,43,1], id_to_word_s)
print ids_to_phrases([234, 432, 102, 12,43,1], id_to_word_t)

from nltk.translate import bleu_score

def nonunique_ngrams(phrase, N):

        N_grams = {}
        for i in range(len(phrase)):
            li = phrase[i:i+N]
            ng = ' '.join([str(s) for s in li])
            if len(li) == N or len(phrase)<N:
                try:
                    N_grams[ng] += 1
                except KeyError:
                    N_grams[ng] = 1
        return N_grams, len(phrase)

def remove_EOS_PAD(long_phrase):
    i=0
    phrase= []
    while (long_phrase+[0])[i] not in (0,1):
        phrase.append(long_phrase[i])
        i+=1
    return phrase

def BLEU_metric(long_t_phrase, long_p_phrase, N):
    # a) Find all (non-unique) N grams in target and predicted phrase and frequencies
    # Firstly need to see how long the content is (not <EOS> or <PAD>)
    t_phrase = remove_EOS_PAD(long_t_phrase)
    p_phrase = remove_EOS_PAD(long_p_phrase)
    N = min(N, len(p_phrase), len(t_phrase))
    t_ngrams, t_len = nonunique_ngrams(t_phrase, N)
    p_ngrams, p_len = nonunique_ngrams(p_phrase, N)

    #print "N-gram count is {}".format(p_ngrams)
    p_num = sum(p_ngrams.values())

    #print p_num
    # b) How many of the N-grams in the prediction appear in the target + frequencies
    # d) Limit the number of correct counts of an Ngram to
    #    the number of times it appears in the target
    cross_count = []
    for ng in p_ngrams.keys():
        try:
            cross_count.append(min((t_ngrams[ng], p_ngrams[ng])))
        except KeyError:
            cross_count.append(0)
    # e) return the above number divided by the total number of (non-unique) N-grams
    # I take the log of the BLEU scores so I can sum them
    # and exponentiate to calculate the product (for geometric mean later on)
    #print float(p_num)
    return [np.log(sum(cross_count)/float(p_num)), t_len, p_len]

# Test handwritten BLEU
p_phrase1 = [4,5,4,5,4,5, 1, 0]
t_phrase = [4,5,6,34,8,76, 87, 1]
assert len(t_phrase) == len(p_phrase1)
t_phrase = remove_EOS_PAD(t_phrase)
p_phrase1 = remove_EOS_PAD(p_phrase1)

print "BLEU1 score test is {}.".format(
    bleu_score.corpus_bleu([[t_phrase]], [remove_EOS_PAD(p_phrase1)], weights=([1])))
print "BLEU2 score test is {}.".format(
    bleu_score.corpus_bleu([[t_phrase]], [remove_EOS_PAD(p_phrase1)], weights=([0.5,0.5])))


source_train = load_obj("source_train")
target_train = load_obj("target_train")
# Obviously vocab preprocessing doesn't get to see the test data
source_test = load_obj("source_test")
target_test = load_obj("target_test")




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

embed_vocab_s, skipped_s = get_embeddings(id_to_word_s, lang=lang_s)
embed_vocab_t, skipped_t = get_embeddings(id_to_word_t, lang=lang_t)

test_x = [[5,2,3],[2], [4,2], [1,2]]
# it's going to go from the number of cols being the sequence length/ num of rows being batch size
# to the number of rows being the max sequence length/ num cols being batch size
# Essentially like a padding and then transpose
def format_batch(x):
    seq_lengths = [len(row) for row in x]
    n_batches = len(x)
    max_seq_length = max(seq_lengths)
    outputs = np.zeros(shape=(max_seq_length, n_batches),dtype=np.int32)
    for i in range(len(seq_lengths)):
        for j in range(seq_lengths[i]):
            outputs[j][i] = x[i][j]
    return outputs

print format_batch(test_x)
print np.array(format_batch(train_t[0:7]))


import tensorflow as tf
tf.reset_default_graph()
sess = tf.InteractiveSession()
input_embedding_size = 64 # Fixed due to pretrained embedding files
encoder_hidden_units = 256
decoder_hidden_units = encoder_hidden_units # Must be the same at the moment

encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
#print encoder_inputs.shape
encoder_inputs_embedded = tf.nn.embedding_lookup(embed_vocab_s, encoder_inputs)
decoder_inputs_embedded = tf.nn.embedding_lookup(embed_vocab_t, decoder_inputs)
#print encoder_inputs_embedded.shape

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

#decoder_logits_2 = tf.contrib.layers.linear(decoder_outputs, vocab_size_t)
#print decoder_logits_2
decoder_prediction = tf.argmax(decoder_logits, axis=2)

timestep_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size_t, dtype=tf.float32),
    logits=decoder_logits,)

# loss is the mean of the cross entropy
loss = tf.reduce_mean(timestep_cross_entropy)

# We use AdaM which combines AdaGrad (parameters updated less often get updated more strongly)
# and momentum (updates depend on the slope of previous updates - avoiding local minima)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())
# Test format_batch and make sure that the decoder
# and encoder accepts inputs with a forward pass

batch_ = [[2,124,243], [24,523,23], [9, 82]]

batch_ = format_batch(batch_)
print('batch_encoded:\n' + str(batch_))

din_ = format_batch(np.ones(shape=(3, 4), dtype=np.int32))
print('decoder inputs:\n' + str(din_))

pred_ = sess.run(decoder_prediction,
    feed_dict={
        encoder_inputs: batch_,
        decoder_inputs: din_,
    })
print('decoder predictions:\n' + str(pred_))

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


# Test everything is working okay

batch_size = 100

for s_sample_batch, t_sample_batch in batch_source_target(train_s[0:2], train_t[0:2], batch_size):
    fd_keys = [encoder_inputs, decoder_inputs, decoder_targets]
    fd = make_feed_dict(fd_keys, s_sample_batch, t_sample_batch)
    fd_r = make_feed_dict(fd_keys, s_sample_batch, t_sample_batch, reverse_encoder_inputs= True)
    fd_t = make_test_feed_dict(fd_keys, s_sample_batch, t_sample_batch, reverse_encoder_inputs= False)
    assert len(fd.values()[0].T[0]) == len(fd_r.values()[0]) # reversed list must be the same length as original
    print fd.keys()[0]
    print np.array(fd.values()[0]).T[0]
    print "Reversed as in Sutskever et al. "
    print np.array(fd_r.values()[0]).T[0]
    assert len(fd.values()[1].T[0]) == len(fd.values()[1].T[1]) # decoder inputs and targets must be the same

    for i in range(len(fd.keys())-1):
        print fd.keys()[i+1]
        print np.array(fd.values()[i+1]).T[0]

    print "Decoder inputs at test time"
    print np.array(fd_t.values()[1]).T[0]
    break

loss_track = []

def format_idx(idx):
    # Just cuts out the padding of word index lists
    li = []
    for i in idx:
        if i ==0:
            break
        else:
            li.append(i)
    return li

BLEU = []
epochs = 30 # How many times we loop over the whole training data
batch_size = 92 # After how many sequences do we update the weights?
print "there will be {} samples in the final batch".format(len(train_s)%batch_size)
fd_keys = [encoder_inputs, decoder_inputs, decoder_targets]
try:
    batch_n = 0
    print "training has begun..."
    for epoch in range(epochs):
        for s_batch, t_batch in batch_source_target(train_s, train_t, batch_size):
            feed_dict = make_feed_dict(fd_keys, s_batch, t_batch)
            _, l = sess.run([train_op, loss], feed_dict)

            #if batch_n == 0 or batch_n == 60:
            #    batch_n += 1
            batch_n +=1
        loss_track.append(l)
        print "epoch {}".format(epoch+1)
        print 'batch {}'.format(batch_n)
        print 'loss: {}'.format(sess.run(loss, feed_dict))
        predict_ = sess.run(decoder_prediction, feed_dict)
        #predictions = [remove_EOS_PAD(pred) for pred in predict_.T]
        #actuals = [[remove_EOS_PAD(act)] for act in fd[decoder_targets].T]
        #BLEU2 = bleu_score.corpus_bleu(actuals, predictions, weights=(0.5,0.5))
        #BLEU.append(BLEU2)
        for (inp, act, pred) in zip(feed_dict[encoder_inputs].T,
                                                 feed_dict[decoder_targets].T,
                                                 predict_.T):
            print '  sample {}:'.format(i + 1)
            print '    input     > {} \n {}'.format(format_idx(inp), ' '.join(ids_to_phrases(inp, lang=lang_s)))
            #)
            print '    actual     > {} \n {}'.format(format_idx(act), ' '.join(ids_to_phrases(act, lang=lang_s)))
            print '    predicted     > {} \n {}'.format(format_idx(pred), ' '.join(ids_to_phrases(inp, lang=lang_s)))


    print 'Training is complete'
except KeyboardInterrupt:
    print 'training interrupted'

save_obj(loss_track, "loss_track")
#plt.plot(, loss_track)
#l = [s for i,s in sorted(zip([len(row) for row in l], l))]
