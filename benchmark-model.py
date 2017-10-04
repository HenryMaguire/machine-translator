from utils import load_obj, save_obj

raw_source_test = load_obj("DATA/preprocess/raw_source_test")
raw_source_test = [seq.split(" ") for seq in raw_source_test]
lang_t = 'en'

from googletrans import Translator


def googletrans_pass(s_text, target_lang='fr'):
    translator = Translator()
    trans_corpus = []
    skipped_phrases = []
    for i, phrase in s_text:
        try:
            trans_corpus.append((i, [trans.text for
                            trans in translator.translate(phrase, dest=target_lang)]))
        except ValueError as err:
            # Making a new Translator instance seems to help JSON errors
            translator = Translator()
            skipped_phrases.append((i, phrase))
            print "{} for phrase {}".format(err, i)
    return trans_corpus, skipped_phrases

def translate_word_by_word(source_text):
    # Need to keep track of phrase ordering by labelling
    source_text = [(i, phr) for i, phr in enumerate(source_text)]
    trans_text, skipped_phrases = googletrans_pass(source_text, target_lang=lang_t)
    print "There are {} phrases which could not be translated first time around.".format(
                                                                    len(skipped_phrases))
    # Keep making passes until there are no more untranslated phrases left
    j = 2
    while len(skipped_phrases)>0:
        translated_corpus = []
        tc_s, skipped_phrases = googletrans_pass(skipped_phrases, target_lang=lang_t)
        trans_text += tc_s
        print "There are {} phrases which could not be translated in pass {}.".format(
                                                            len(skipped_phrases), j)
        j+=1
    # Sort the phrases via their labels.
    # Sorted function gives ([indices], [phrases]) so just need 2nd element
    trans_text = zip(*sorted(zip(*BM_translated)))[1]
    return trans_text

BM_translated = translate_word_by_word(raw_source_test)
print "Corpus translated from {} to {}".format(lang_s, lang_t)
save_obj(BM_translated, "BM_translated_test")
