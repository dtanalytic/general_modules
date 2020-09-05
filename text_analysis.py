import nltk
from string import punctuation
from nltk.stem import SnowballStemmer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def word_filter_filedelete(list_words,filedeletename, encoding='utf-8'):
    list_delete = []
    whole_word_filter=True
    if len(filedeletename)==2:
        whole_word_filter=filedeletename[1]
        filedeletename = filedeletename[0]
        
    with open(filedeletename,encoding=encoding) as f:
        for line in f:
            list_delete.append(line.strip())
    if whole_word_filter:
        list_words_new = [w for w in list_words if not w in list_delete]
    else:
        list_words_new = list_words.copy()# если из самого list_words удалять порядок элементов сбивается
        for w in list_words:
            for w_d in list_delete:
                if w.find(w_d)!=-1:
                    list_words_new.remove(w)
                    break
    
    return list_words_new


def _word_stemmer_dict(list_words,stemmer,dict_stem):
    new_word_list = []
    for item in list_words:
        stem_word = stemmer.stem(item) if not item.istitle() else stemmer.stem(item).capitalize()
        if stem_word in dict_stem.index:
            new_word_list.append(_find_stem_word_dict(dict_stem, stem_word, item))
        elif stem_word.istitle() and stem_word.lower() in dict_stem.index:
            new_word_list.append(_find_stem_word_dict(dict_stem, stem_word.lower(), item.lower()))
        else:
            new_word_list.append(item)
                
    return new_word_list


def _find_stem_word_dict(dict_stem, stem_word, item):
    df_temp=dict_stem.loc[stem_word]
    if isinstance(df_temp,pd.DataFrame):
        for i in range(len(df_temp)):
            if item in df_temp.iloc[i]['формы'].split(':'):
                    return df_temp.iloc[i]['инфинитив']
            
    elif isinstance(df_temp,pd.Series):            
        if item in df_temp['формы'].split(':'):
                       
            return df_temp['инфинитив']
            
    return item



def preprocess_text(text, nltk_stop_lang='russian', nltk_stemmer_lang='russian', min_len_token=3, 
                    stop_words_files=['../general_modules/словари/delete.txt','../general_modules/словари/delete_sym.txt',('словари/stops.txt',1)], 
                    dict_stem_file='../general_modules/словари/морфологический словарь.txt', to_lower=True, stem_flag=True, save_sents=False):
    #text=text.lower()
    
    list_words = nltk.tokenize.word_tokenize(text)
    list_words = preprocess_list(list_words,to_lower,nltk_stop_lang,nltk_stemmer_lang, min_len_token, stop_words_files, dict_stem_file,stem_flag)
    if save_sents:
       return ' '.join(list_words)+'.'
    else:
       return ' '.join(list_words)


# файлы stop_words_files должны содержать по одному исключаемому слову в строчке
def preprocess_list(list_words, to_lower=True, nltk_stop_lang='russian', nltk_stemmer_lang='russian', min_len_token=3, stop_words_files=['словари/delete.txt','словари/delete_sym.txt',('словари/stops.txt',1)],
                    dict_stem_file='словари/морфологический словарь.txt', stem_flag=True):

    punct=list(punctuation)
    list_words = [w for w in list_words if not w.isdigit() and not w in punct]

    if nltk_stop_lang:
        stop_words = nltk.corpus.stopwords.words(nltk_stop_lang)
        list_words = [w for w in list_words if not w in stop_words]
    if min_len_token:
        list_words = [w for w in list_words if len(w)>=min_len_token]
    if stop_words_files:
        for filename in stop_words_files:
            list_words = word_filter_filedelete(list_words, filename)

    if stem_flag:
        #list_words = word_stemmer(list_words, dict_stem_file, nltk_stop_lang)
        if isinstance(dict_stem_file,str):
            dict_stem = pd.read_csv(dict_stem_file, sep='|', encoding='cp1251')
            dict_stem = dict_stem.dropna()
            dict_stem = dict_stem.set_index('стем')
        else:
            dict_stem = dict_stem_file
        stemmer = SnowballStemmer(nltk_stemmer_lang)
        list_words = _word_stemmer_dict(list_words,stemmer,dict_stem)
        # при первом запуске непростемменизированные стоп слова могли остаться... 
        # но в начале эти же действия оставляем, чтобы меньше слов стемить
        if nltk_stop_lang:
            list_words = [w for w in list_words if not w in stop_words]
        if stop_words_files:
            for filename in stop_words_files:
                list_words = word_filter_filedelete(list_words, filename)
    if to_lower:
        list_words = [w.lower() for w in list_words]
            
    return list_words


def set_allowed_symbols_token(token, allowed_pattern, repl_symb='?'):
    list_symbs = []
    for symb in token:
        list_symbs.append(symb) if allowed_pattern.search(symb) else list_symbs.append(repl_symb)
    return ''.join(list_symbs)


# класс удобен для анализа частот одиночных файлов, повторяет работу preprocess_list
# stopwords stop_words='russian' - stop_words реализована в классе, но вторым прогоном сами делаем
# пунктуация откидывается автоматически
class StemmedCountVectorizer(CountVectorizer):

    def __init__(self, stemmer,nltk_stop_lang='russian',min_len_token=3,files_delete=['словари/delete.txt','словари/delete_sym.txt',('словари/stops.txt',1)],
                 file_dict='словари/морфологический словарь.txt', 
                 input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=False, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word',
                 max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None, binary=False, dtype=np.int64):


        self.stemmer = stemmer
        self.min_len_token = min_len_token
        self.files_delete=files_delete
        self.nltk_stop_lang = nltk_stop_lang
        
        if nltk_stop_lang:
            stop_words = nltk.corpus.stopwords.words(nltk_stop_lang)
            self.stop_words  = stop_words
        self.encoding=encoding
        self.file_dict = file_dict
        self.df_dict= pd.read_csv(file_dict, sep='|', encoding='cp1251')
        self.df_dict=self.df_dict.dropna()
        self.df_dict = self.df_dict.set_index('стем')
        super().__init__(input, encoding,
                 decode_error, strip_accents,
                 lowercase, preprocessor, tokenizer,
                 stop_words, token_pattern,
                 ngram_range, analyzer,
                 max_df, min_df, max_features,
                 vocabulary, binary, dtype)

    def preprocess(self,analyzer,doc):
        try:

            list_words = [w for w in analyzer(doc) if not w.isdigit()]
            if self.min_len_token:
                list_words = [w for w in list_words if len(w)>=self.min_len_token]
           
            if self.files_delete:
                for filename in self.files_delete:
                    list_words = word_filter_filedelete(list_words, filename,self.encoding)
            
            list_words = _word_stemmer_dict(list_words, self.stemmer, self.df_dict)
            list_words = [w.lower() for w in list_words]
            
       
            # при первом запуске непростемменизированные стоп слова могли остаться... 
            # но в начале эти же действия оставляем, чтобы меньше слов стемить
            if self.nltk_stop_lang:
                list_words = [w for w in list_words if not w in self.stop_words]
            if self.files_delete:
                for filename in self.files_delete:
                    list_words = word_filter_filedelete(list_words, filename)
            
            return list_words


        except Exception as e:
              print(e)

    def build_analyzer(self):
        analyzer=super(StemmedCountVectorizer,self).build_analyzer()

        return lambda doc: self.preprocess(analyzer,doc)



if __name__== '__main__':
    import sys
    sys.path.append('../')    
    from general_modules.text_analysis import StemmedCountVectorizer
    from general_modules.text_stats import words_counts
    
    text='по жили Были, S ... -- квартире " ,<Сирии, Сирия 1912 два кота и дочка\nкрасивая-такая\nкоторого'
    text=preprocess_text(text, nltk_stop_lang = None, 
                         nltk_stemmer_lang = 'russian',stop_words_files = ['../general_modules/словари/delete_sym.txt'], 
                         min_len_token = None,stem_flag=True)


 
    # vectorizer = StemmedCountVectorizer(SnowballStemmer('russian'), min_len_token=3, 
    #                                     files_delete=['словари/delete.txt',('словари/spec.txt',0),'словари/delete_sym.txt', ('словари/stops.txt',1)]
    #                                     )
    # freq_dict = words_counts(text,vectorizer)
 
