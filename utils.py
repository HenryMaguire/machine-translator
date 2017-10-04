
import pickle
import traceback
def load_obj(name):
    with open(name + '.pickle', 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name ):
    with open(name + '.pickle', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def ids_to_phrases(idx_list, id_to_word):
    # Takes list of word ids and returns a string of words
    # Mainly for use in analysis
    phrase = u''
    i=0
    if len(idx_list)>0:
        "We want to ignore <EOS> unless it's the first element"
        while (i>=0) and (idx_list[i] not in (1,0)):
            phrase+= id_to_word[idx_list[i]]+u' '
            i+=1
    return phrase.encode("utf-8")

def remove_EOS_PAD(long_phrase):
    """ Formats phrases by removing EOS
    and PAD for readability
    """
    i=0
    phrase= []
    while (long_phrase+[0])[i] not in (0,1):
        phrase.append(long_phrase[i])
        i+=1
    return phrase

def format_batch(x):
    """ Embeds a list of lists into a matrix
    padding the ends of seqs with zeros
    """
    seq_lengths = [len(row) for row in x]
    n_batches = len(x)
    max_seq_length = max(seq_lengths)
    outputs = np.zeros(shape=(max_seq_length, n_batches),dtype=np.int32)
    for i in range(len(seq_lengths)):
        for j in range(seq_lengths[i]):
            outputs[j][i] = x[i][j]
    return outputs

def format_idx(idx):
    """ Just cuts out the padding of word index lists
    """
    li = []
    for i in idx:
        if i ==0:
            break
        else:
            li.append(i)
    return li
