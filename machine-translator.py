import codecs
import numpy as np
f_fr = codecs.open("DATA/en-fr_paropt/dev.tok.fr", encoding='utf-8')
f_en = codecs.open("DATA/en-fr_paropt/dev.tok.en", encoding='utf-8')


lang_t = 'en'
lang_s = 'fr'

raw_source = eval("f_"+lang_s).read()
raw_target = eval("f_"+lang_t).read()


#print raw_source[0:1000]
print "Corpora aligned: {}".format(len(raw_source.split('\n')) == len(raw_target.split('\n')))


import nltk.data
import random
# First split up into lines using the Punkt tokenizer
#fr_sent_detector = nltk.data.load('tokenizers/punkt/french.pickle')
#en_sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
#fr_sents = fr_sent_detector.tokenize(text_fr)
#en_sents = en_sent_detector.tokenize(text_en)

# Split the text up into lines
# Randomise the lists but maintain parallel ordering
s = zip(raw_source.split('\n'), raw_target.split('\n'))
np.random.shuffle(s)
source_phrases, target_phrases = zip(*s)
"""
print "\n".join([source_phrases[0],target_phrases[0]])
print "------------"
print "\n".join([source_phrases[302],target_phrases[302]])
print "------------\n Judging by keywords and rudimentary knowledge of French, I suspect the corpora are still aligned."
print "corpora still same length:", len(source_phrases)== len(target_phrases), '\n'
"""
