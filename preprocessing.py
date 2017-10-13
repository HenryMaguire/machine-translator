import codecs
import numpy as np
from utils import *
import nltk.data
import random
import nltk
from multiprocessing import Pool
import multiprocessing as mp
import string
from sklearn.cross_validation import train_test_split

def remove_punc(text):
    # Needs to keep stops like L.A.P.D.
    # almost all other punctuation needs to be carefully removed
    text = unicode.strip(text)
    text = unicode.replace(text, " .", u"")
    text = unicode.replace(text,",", u"")
    text = unicode.replace(text,u" .", u"")
    text = unicode.replace(text,u" ,", u"")
    text = unicode.strip(text, ",:)(][}{!")
    text = unicode.strip(text, u",:)(][}{;!")
    text = unicode.replace(text,u':', u'')
    text = text.replace(u';', u'')
    text = text.replace(')', u'')
    text = text.replace("(", u'')
    text = text.replace("!", u':')
    #text = text.replace(u"     ", " ")
    text = text.replace(u"    ", " ")
    text = text.replace(u"   ", " ")
    text = text.replace(u"  ", " ")
    return text

lang_t = 'en'
lang_s = 'fr'
prep_path = "/DATA/"

f1_fr = codecs.open("DATA/en-fr_paropt/dev.tok.fr", encoding='utf-8')
f2_fr = codecs.open("DATA/en-fr_paropt/train.fr", encoding='utf-8')
#test_fr = codecs.open("DATA/en-fr_paropt/test.fr", encoding='utf-8')
source = remove_punc(eval("f1_"+lang_s).read()+"\n"+eval("f2_"+lang_s).read())


f1_en = codecs.open("DATA/en-fr_paropt/dev.tok.en", encoding='utf-8')
f2_en = codecs.open("DATA/en-fr_paropt/train.en", encoding='utf-8')
#test_en = codecs.open("DATA/en-fr_paropt/test.en", encoding='utf-8')
target = remove_punc(eval("f1_"+lang_t).read()+"\n"+eval("f2_"+lang_t).read())


#i = source.index(u'pr\xe9sident m. rohan')
#print i, source[i:i+60]


print "Length of both corpora: {}, {}".format(len(source.split('\n')), len(
                                                        target.split('\n')))

del f1_fr, f2_fr, f1_en, f2_en


import nltk
def tokenize_sentences(text_source, text_target, max_source_length=23):
    new_text_s = []
    new_text_t = []
    flat_s = []
    flat_t = []
    for sent_s, sent_t in zip(text_source, text_target):
        # This splits up tokens within a sentence and removes whitespace from the ends
        tok_s = (unicode.strip(sent_s.lower())).split(' ')
        tok_t = (unicode.strip(sent_t.lower())).split(' ')
        bool_short_s = (len(tok_s)<=max_source_length)
        bool_short_t = (len(tok_t)<=(max_source_length-4))
        bool_asymm = abs(len(tok_t)-len(tok_s))<int(max_source_length/2)
        if bool_short_s and bool_short_t and bool_asymm:
            flat_s += tok_s
            flat_t += tok_t
            # I'm keeping the final punctuation and appending the
            # <EOS> tag after it
            new_text_s.append(tok_s)
            new_text_t.append(tok_t)
    print "There are now {} sequence pairs remaining.".format(len(new_text_t))
    print "Corpora still same length: {}".format(len(new_text_t)==len(new_text_s))
    return new_text_s, nltk.FreqDist(flat_s), new_text_t, nltk.FreqDist(flat_t)


test_fraction = 0.05 # percentage of data to use as test


source_train, source_test, target_train, target_test = train_test_split(
                                source.split('\n'), target.split('\n'),
                                test_size=test_fraction, random_state=42)

print "TESTING DATA"
source_test, freq_s, target_test, freq_t = tokenize_sentences(source_test, target_test)
print "TRAINING DATA"
source_train, freq_s, target_train, freq_t = tokenize_sentences(source_train, target_train)


"""
def further_train_cleaning(source, target):
    # Further clean training data by removing data which differ in size by
    # too much. These cases tend to be false data.
    # Cannot have access to this info for test data.
    o_s, o_t = [], []
    print zip(source, target)[10:20]
    for s, t in zip(source, target):
        if abs(len(t)-len(s))<12 or (len(t)>22):
            o_s.append(s)
            o_t.append(t)
    return o_s, o_t
"""
#source_train, target_train = further_train_cleaning(source_train, target_train)
# We need to save the raw test data to file to use for the benchmark model
print max([len(s) for s in source_train])
print max([len(s) for s in target_train])
print "Training and test splitting and tokenising complete"
del target, source

print "\n".join([' '.join(source_train[1]), ' '.join(target_train[1])])
print "------------"
print "\n".join([' '.join(source_train[10]), ' '.join(target_train[10])])
print "------------\n Judging by keywords and rudimentary knowledge of French, are the corpora are still aligned?"
print "corpora still same length with training length: {} and testing: {}".format(
                                            len(source_train),len(source_test))

#save_obj(source_test_, "DATA/source_test") # For benchmark model
#save_obj(target_test_, "DATA/target_test")

def word_to_ids(freq_dist, vocab_size, lang):
    total_words = len(freq_dist.keys())
    total_length = sum(freq_dist.values())
    print "{} vocab size restricts to {} percent of total vocab.".format(lang, 100*(float(vocab_size)/total_words))
    vocab = dict(freq_dist.most_common(vocab_size))
    print "vocab size= ",len(vocab.keys())
    covered_with_vocab = sum(vocab.values()) # Total of frequencies
    print "Percentage of whole {} corpus covered by vocab: {}".format(lang,
                                            100*(covered_with_vocab/float(total_length)))
    # IDs begin at 3 because <EOS>=1 and <UNK>=2
    word_to_ids = dict([(word, i+3) for i, word in enumerate(vocab.keys())])
    word_to_ids[u'<UNK>'] = 2
    word_to_ids[u'<EOS>'] = 1
    word_to_ids[u'<PAD>'] = 0
    id_to_words = dict((idx, word) for word, idx in word_to_ids.items())
    return word_to_ids, id_to_words

vocab_size_s = 25000
vocab_size_t = 20000

#word_to_ids(freq_s_d, vocab_size_s_d, lang_s)
print "Source dataset"
word2id_s, id2word_s =  word_to_ids(freq_s, vocab_size_s, lang_s)
n = 60

print "{} most common source words: {}".format(n, freq_s.most_common(n))
print "Full dataset"
word2id_t, id2word_t = word_to_ids(freq_t, vocab_size_t, lang_t)
print "{} most common source words: {}".format(n, freq_t.most_common(n))
print id2word_t[5], word2id_t[id2word_t[5]]
print "S and T Vocabulary sizes: {} and {}".format(len(word2id_s.keys()),
                                                    len(word2id_t.keys()))
import time
from functools import partial

def replace_id_func(sequence, **kwargs):
    word2id=kwargs['wtid']
    ids_sent = [] # sentence with words replaced by ids
    sequence+=[u'<EOS>']
    for token in sequence:
        if token not in word2id.keys():
            if token == u'<EOS>':
                ids_sent.append(1)
            else:
                ids_sent.append(2)
        else:
            ids_sent.append(word2id[token])
    return ids_sent

def replace_with_word_id(text, wtid, lang):
    '''
    take the list of lists, find FreqDist, replace any
    out of vocabulary words with <UNK> whilst giving each
    token a numerical ID, return new list of lists'''
    ti = time.time()
    prep_text = []
    pool = Pool(processes = 3)
    kwargs = {'wtid': wtid}
    id_text = [i for i in pool.imap(partial(replace_id_func, **kwargs), text, chunksize=20000)]
    print "Replacing words with IDs took {} seconds.".format(time.time()-ti)
    return id_text

source_train = replace_with_word_id(source_train, word2id_s, lang_s)

target_train = replace_with_word_id(target_train, word2id_t, lang_t)
source_test = replace_with_word_id(source_test, word2id_s, lang_s)
target_test = replace_with_word_id(target_test, word2id_t, lang_t)


data_dic = {"s_train": source_train, "t_train": target_train,
                "s_test": source_test, "t_test": target_test,
                "word2id_s": word2id_s, "word2id_t": word2id_t,
                "id2word_s": id2word_s, "id2word_t": id2word_t}
save_obj(data_dic, "DATA/data_dic")

def create_dev_data(id_text, id2word, vocab_size):
    id2word = dict(sorted(id2word.items())[0:vocab_size])
    word2id = dict((word, idx) for idx, word in id2word.items())
    simple_text = []
    for phrase in id_text:
        p = []
        for token in phrase:
            if token in id2word.keys():
                p.append(token)
            else:
                p.append(2)
        simple_text.append(p)
    return simple_text, word2id, id2word

# Need a very small debugging dataset

s_train_dev, word2id_s_dev, id2word_s_dev = create_dev_data(source_train[0:2100], id2word_s, 1000)
t_train_dev, word2id_t_dev, id2word_t_dev = create_dev_data(target_train[0:2100], id2word_t, 1000)
s_train_dev = s_train_dev[0:2000]
t_train_dev = t_train_dev[0:2000]
t_test_dev = t_train_dev[2000:2100]
s_test_dev = s_train_dev[2000:2100]
dev_dic = {"s_train": s_train_dev, "t_train": t_train_dev,
            "s_test": s_test_dev, "t_test": t_test_dev,
            "word2id_s": word2id_s_dev, "word2id_t": word2id_t_dev,
            "id2word_s":id2word_s_dev, "id2word_t": id2word_t_dev
            }
save_obj(dev_dic, "DATA/dev_dic")
