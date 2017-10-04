from utils import *

word2id_s = load_obj("DATA/preprocess/word2id_s")
word2id_t = load_obj("DATA/preprocess/word2id_t")
id2word_s = load_obj("DATA/preprocess/id2word_s")
id2word_t = load_obj("DATA/preprocess/id2word_t")

text_source = load_obj("DATA/preprocess/source_train") # Preprocessed already
text_target = load_obj("DATA/preprocess/target_train") # Preprocessed already

vocab_size_t = len(word2id_t.keys())
vocab_size_s = len(word2id_s.keys())
lang_s = 'fr'
lang_t = 'en'

import pickle
import numpy as np
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
