import sys
sys.path.append('../')

from general_modules.reader import CorpusReader
from general_modules.multi_magic import MultiFilesOperator

from itertools import dropwhile,takewhile
from collections import defaultdict
from nltk import FreqDist
from collections import Counter
import os

def corpus_info(file_id,files_encoding,ignore_digits=True,ignore_punct=True):

    path_to_load = os.path.dirname(file_id)
    file_id = os.path.basename(file_id)
    reader = CorpusReader(input_folder_name=path_to_load,\
                          doc_pattern=r'(.*?/).*\.txt', categ_pattern=r'(.*?)/.*\.txt',
                          encoding=files_encoding)

    
    return {'paras_list':[item for item in list(reader.paras(fileids=[file_id])) if item!=''],
            'sents_list':list(reader.sents(fileids=[file_id])),
            'words_list':list(reader.words(fileids=[file_id], ignore_digits=ignore_digits,ignore_punct=ignore_punct )),
            }


def uni_structs(dicts_list, len_w_big=7):
    
    uni_dict = defaultdict(list)
    for key in dicts_list[0].keys():
        for dict in dicts_list:
            uni_dict[key].extend(dict[key])
    
    statsdict = texts_features(uni_dict['paras_list'], uni_dict['sents_list'],uni_dict['words_list'], len_w_big=len_w_big)

    return statsdict,uni_dict

# выводит количественные статистики для наборов, предлоежений, абзацев и слов
def texts_features(paras_list, sents_list,words_list, len_w_big=7,print_results=False):
    
    num_paras = len(paras_list)
    num_sents = len(sents_list)
    sperp = num_sents / num_paras
    tokens = FreqDist(words_list)
    words_count = sum(tokens.values())
    vocab = len(tokens)
    lexdiv = words_count/vocab    
    words_big = [word for word in words_list if len(word)>len_w_big]
    w_big_num = len(words_big)
    
    # flash_k_ind = 0.4*(0.78*words_count/num_sents + 100*w_big_num/words_count)
    flash_k_ind = 0.4*(words_count/num_sents + 100*w_big_num/words_count)

    if print_results:
        print((
            "Текст сформирован из {} параграфов и {} предложений.\n"
            "{:0.3f} предложений на параграф \n"
            "Всего слов {} при словаре из {} уникальных слов\n"
            "Лексическое разнообразие -{:0.3f}\n"
            "Индекс туманности - {:0.3f}\n"
        ).format(num_paras, num_sents, sperp, words_count, vocab, lexdiv,flash_k_ind
        ))
            
    statsdict={'num_paras':num_paras,'num_sents':num_sents,\
               'sperp':sperp,'words_count':words_count,'vocab':vocab,'words{}+'.format(len_w_big):w_big_num,'lexdiv':lexdiv,'flash_k_ind':flash_k_ind}

    return statsdict
    

def words_counts(string,vectorizer):
    matrix_counts = vectorizer.fit_transform([string])
    matrix_counts = matrix_counts.toarray()
    
    freq_dict = {name:matrix_counts[0][i] for i,name in enumerate(vectorizer.get_feature_names())}
    
    return freq_dict

#crops dict according to absolute values or percents of freq of max_freq 
def crop_freq_dict(freq_dict, per_min,per_max):
    
    word_counter_dict = Counter(freq_dict)        
    if isinstance(per_min, float) and isinstance(per_max, float): 
        x_max = word_counter_dict.most_common()[0][1]*per_max if word_counter_dict.most_common()[0][1]*per_max < word_counter_dict.most_common()[0][1]-1 else word_counter_dict.most_common()[0][1]-1 
        x_min = word_counter_dict.most_common()[0][1]*per_min  if word_counter_dict.most_common()[0][1]*per_min > 1 else 1
    else:
        x_max = per_max
        x_min = per_min
    # возвращает элементы начиная с первого для которого функция lambda - False, то есть получим и удалим редкие
    for key, count in dropwhile(lambda key_count: key_count[1] > x_min, word_counter_dict.most_common()):
        del word_counter_dict[key]
    # возвращает элементы до первого элемента где функция будет False, то есть получим и удалим частые
    for key, count in takewhile(lambda key_count: key_count[1] > x_max, word_counter_dict.most_common()):
        del word_counter_dict[key]
        
    return word_counter_dict, (x_min,x_max)
    
    return 



if __name__=='__main__':
    
    # from nltk.stem import SnowballStemmer
    # import sys
    # sys.path.append('../')
    
    # from general_modules.text_analysis import StemmedCountVectorizer
    # from general_modules.text_stats import words_counts
        
    # vectorizer = StemmedCountVectorizer(SnowballStemmer('russian'), min_len_token=3)
    # text = 'я была в авдотья водянка, россией, горы горах, горах Путина шило мыло ширли твирли твирли была я и никуда не хочу'
    # freq_dict = words_counts(text,vectorizer)     


    statdict1 = texts_features(**corpus_info('input_texts/temp3/export 2019-12-16 1175.txt', 'utf8' ))
    mfo = MultiFilesOperator(process_num=6)
    results = mfo.multi_files_operator(corpus_info,path_to_load='input_texts/temp3',\
                                        files_encoding='utf8')
        
    statsdict2,uni_dict = uni_structs(results,len_w_big=7)

    
    
    
    