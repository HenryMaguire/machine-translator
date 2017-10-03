import codecs
import numpy as np
from utils import *
import nltk.data
import random
import nltk
from multiprocessing import Pool
def tokenize_sentences(text):
    new_text = []
    flat = []
    for sentence in text:
        # This splits up tokens within a sentence
        tok_sent = (sentence.lower()).split(' ')
        flat += tok_sent
        # I'm keeping the final punctuation and appending the
        # <EOS> tag after it
        new_text.append(tok_sent+[u'<EOS>'])
    freq = nltk.FreqDist(flat)
    return new_text, freq


def word_to_ids(freq_dist, vocab_size, lang):
    total_words = len(freq_dist.keys())
    total_length = sum(freq_dist.values())
    print "{} vocab size restricts to {} percent of total vocab.".format(lang, 100*(float(vocab_size)/total_words))
    vocab = dict(freq_dist.most_common(vocab_size+1))
    covered_with_vocab = sum(vocab.values()) # Total of frequencies
    print "Percentage of whole {} corpus covered by vocab: {}".format(lang,
                                            100*(covered_with_vocab/float(total_length)))
    # The identities begin at 3, since <EOS>=1 and <UNK>=2
    word_to_ids = dict([(word, i+3) for i, word in enumerate(vocab.keys())])
    word_to_ids[u'<UNK>'] = 2
    word_to_ids[u'<EOS>'] = 1
    word_to_ids[u'<PAD>'] = 0
    id_to_words = {}
    for word, idx in word_to_ids.items():
        id_to_words[idx] = word
    return word_to_ids, id_to_words

def replace_id_func(sequence):
    ids_sent = [] # sentence with words replaced by ids
    for token in sequence:
        if token not in word_to_ids.keys():
            if token == '<EOS>':
                ids_sent.append(1)
            else:
                ids_sent.append(2)
        else:
            ids_sent.append(word_to_ids[token])
    return ids_sent

def replace_with_word_id(text, word_to_ids, lang):
    '''
    take the list of lists, find FreqDist, replace any
    out of vocabulary words with <UNK> whilst giving each
    token a numerical ID, return new list of lists'''
    prep_text = []
    print len(text)
    pool = Pool()
    pool.map(replace_id_func, text)
    return text


f1_fr = codecs.open("DATA/en-fr_paropt/dev.tok.fr", encoding='utf-8')
f2_fr = codecs.open("DATA/en-fr_paropt/train.fr", encoding='utf-8')
test_fr = codecs.open("DATA/en-fr_paropt/test.fr", encoding='utf-8')

f1_en = codecs.open("DATA/en-fr_paropt/dev.tok.en", encoding='utf-8')
f2_en = codecs.open("DATA/en-fr_paropt/train.en", encoding='utf-8')
test_en = codecs.open("DATA/en-fr_paropt/test.en", encoding='utf-8')

lang_t = 'en'
lang_s = 'fr'

raw_source_train = eval("f1_"+lang_s).read()+"\n"+eval("f2_"+lang_s).read()
raw_target_train = eval("f1_"+lang_t).read()+"\n"+eval("f2_"+lang_t).read()

raw_source_test = eval("test_"+lang_s).read()
raw_target_test = eval("test_"+lang_t).read()

print raw_source_train[0:1000]
print "Corpora aligned: {}".format(len(raw_source_train.split('\n')) == len(raw_target_train.split('\n')))



# First split up into lines using the Punkt tokenizer
#fr_sent_detector = nltk.data.load('tokenizers/punkt/french.pickle')
#en_sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
#fr_sents = fr_sent_detector.tokenize(text_fr)
#en_sents = en_sent_detector.tokenize(text_en)

# Split the text up into lines
# Randomise the lists but maintain parallel ordering
s = zip(raw_source_train.split('\n'), raw_target_train.split('\n'))
np.random.shuffle(s)
raw_source_train, raw_target_train = zip(*s)

print "\n".join([raw_source_train[0],raw_target_train[0]])
print "------------"
print "\n".join([raw_source_train[302],raw_target_train[302]])
print "------------\n Judging by keywords and rudimentary knowledge of French, I suspect the corpora are still aligned."
print "corpora still same length:", len(source_phrases)== len(target_phrases), '\n'

import nltk


tok_source_test, freq_s = tokenize_sentences(raw_source_test)
tok_target_test, freq_t = tokenize_sentences(raw_target_test)

tok_source_train, freq_s = tokenize_sentences(raw_source_train)
tok_target_train, freq_t = tokenize_sentences(raw_target_train)
print "All data tokenised. There are {} sequences for the NMT to learn from.".format(len(tok_source_train))
print tok_source_train[0:3]


vocab_size_s = 6000
vocab_size_t = 6000

#seq_data_source = seq_length_stats(tokenized_source, lang_s)
#seq_data_target = seq_length_stats(tokenized_target, lang_t)

""" For now, sequences form rows of the matrix, with the number of columns
    equal to the maximum length of sequence (or timesteps)"""
vocab_s, id_to_word_s = word_to_ids(freq_s, vocab_size_s, lang_s)
vocab_t, id_to_word_t = word_to_ids(freq_t, vocab_size_t, lang_t)
print id_to_word_s.items()[0:20]

#source_train = replace_with_word_id(tok_source_train, id_to_word_s, lang_s)
#save_obj(source_train, "source_train")
target_train = replace_with_word_id(tok_target_train, id_to_word_t, lang_t)
save_obj(target_train, "target_train")
# Obviously vocab preprocessing doesn't get to see the test data
source_test = replace_with_word_id(tok_source_test, id_to_word_s, lang_s)
save_obj(source_test, "source_test")
target_test = replace_with_word_id(tok_target_test, id_to_word_t, lang_t)
save_obj(target_test, "target_test")
