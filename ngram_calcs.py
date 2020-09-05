import os
import sys
sys.path.append("..")
from general_modules.reader import CorpusReader
from general_modules.stemmer_files import StemmerFiles
from general_modules.util import setup_logger
from general_modules.multi_magic import MultiFilesOperator
import numpy as np
from general_modules.file_operations import split_file_pieces

import pickle
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.probability import FreqDist, ConditionalFreqDist
import configparser
import shutil

import multiprocessing as mp


class NgrammConsolidator():

    def __init__(self, settings_file_name):

        self.logger=setup_logger('consolidator', '%(asctime)s - %(name)s - %(levelname)s - %(message)s',"ngram_logger.log")
        cfg = configparser.ConfigParser()
        cfg.read(settings_file_name)
        self.ngr_num = int(cfg['general']['ngr_num'])
        self.process_num = mp.cpu_count()
        self.results_dir_name = cfg['general']['result_dir_name']
        self.stem_dir_name = cfg['general']['stem_dir_name']

        self.temp_dir_name = cfg['general']['temp_dir_name']
        self.pieces_dir_name = cfg['general']['pieces_dir_name']
        self.ngram_preserver_name = cfg['general']['ngram_preserver_name']
        self.input_stem_dir = os.path.join(self.results_dir_name, self.stem_dir_name)
        self.num_pieces_file_split = cfg['general']['num_pieces_file_split']
        self.paras_delim = cfg['general']['paras_delim']

        if os.path.exists(os.path.dirname(self.results_dir_name) + self.ngram_preserver_name):
            self.ngram_preserver = NgramPreserver(self.ngr_num,  self.ngram_preserver_name)
            self.logger.info('------------\nstarting on base existing lexicon ')
        else:
            self.ngram_preserver = NgramPreserver(self.ngr_num)
    
    def uni_dicts(self, dicts_list):
        allgrams = defaultdict(ConditionalFreqDist)
        ngrams = FreqDist()
        unigrams = FreqDist()

        for dicts in dicts_list:
            for ngram_order in dicts[0].keys():
                for context in dicts[0][ngram_order].keys():
                    for word in dicts[0][ngram_order][context].keys():
                        allgrams[ngram_order][context][word] += dicts[0][ngram_order][context][word]

            for word in dicts[1].keys():
                ngrams[word] += dicts[1][word]
            for word in dicts[2].keys():
                unigrams[word] += dicts[2][word]


        return allgrams, ngrams, unigrams


    def ngram_preserver_from_uni_dicts(self, dicts_list):

        allgrams, ngrams, unigrams = self.uni_dicts(dicts_list)
        ngram_preserver = NgramPreserver(self.ngr_num)
        ngram_preserver.allgrams = allgrams
        ngram_preserver.ngrams = ngrams
        ngram_preserver.unigrams = unigrams
        ngram_preserver.dict_len=len(unigrams)
        return ngram_preserver


    def split_stem(self, filename, encoding,flag_async=True):
        pieces_dir = self.pieces_dir_name + ' ' + os.path.basename(filename).split('.')[0]
        # проверить чтобы в os.path.dir(self.input_filename) – pieces создавал
        split_file_pieces(filename, encoding=encoding, paras_delim = self.paras_delim,\
                          pieces_dir_name = pieces_dir, max_num_paras=self.num_pieces_file_split)
						  
        self.logger.info('------------\nstart stemming file {}'.format(filename))
        
        self.stem(path_to_dir=os.path.join(os.path.dirname(filename),pieces_dir), encoding_files=encoding, flag_async=flag_async)


    def stem(self, path_to_dir, encoding_files,flag_async):
        
        #path_to_dir = os.path.dirname(dirname)
        reader = CorpusReader(input_folder_name=path_to_dir, doc_pattern=r'(.*?/).*\.txt',
                              categ_pattern=r'(.*?)/.*\.txt',
                              encoding=encoding_files)

        
        #fileids = [os.path.join(path_to_dir,item) for item in reader.root_ids]
        fileids = reader.root_ids
        stef = StemmerFiles(path_to_dir, fileids, encoding=encoding_files,nltk_stop_lang = None, nltk_stemmer_lang = 'russian', 
                            dict_stem_file = '../general_modules/словари/морфологический словарь.txt',
                            stop_words_files = ['../general_modules/словари/delete_sym.txt'], min_token_len = None,\
                            results_dir_name=self.results_dir_name, stem_dir_name=self.stem_dir_name,\
                            temp_dir_name=self.temp_dir_name, save_sents=False)

        #stef=StemmerFiles(path_to_dir,fileids,encoding='utf8')
        self.logger.info('-----------\nstart stemming {} file pieces'.format(len(fileids)))
        stef.start_stem_files(flag_async=flag_async)
        

    def load_new_file(self, filename,file_encoding,load_new_top_ngrms='', flag_async=True,filter_ngr_len=0):

        self.logger.info('------------\nstart loading {} file in lexicon'.format(filename))
        self.split_stem(filename, file_encoding,flag_async)
        pieces_dir = self.pieces_dir_name + ' ' + os.path.basename(filename).split('.')[0]
        path_to_load = self.results_dir_name
        # if load_new_top_ngrms:
        #     flag_async=False
        return self.load_new_dir(os.path.join(path_to_load,pieces_dir), file_encoding,load_new_top_ngrms,flag_async,filter_ngr_len)

    def load_new_dir_async(self, path_to_load, files_encoding,load_new_top_ngrms='',filter_ngr_len=0):
        mfo = MultiFilesOperator(process_num=self.process_num)
        results = mfo.multi_files_operator(func_oper=self.start_ngr_calc,path_to_load=path_to_load, files_encoding=files_encoding,load_new_top_ngrms=load_new_top_ngrms, filter_ngr_len=filter_ngr_len)
        return results
    
    def load_new_dir_sync(self, path_to_load, files_encoding,load_new_top_ngrms='', filter_ngr_len=0):
        results=[]
        reader = CorpusReader(input_folder_name=path_to_load,\
                              doc_pattern=r'(.*?/).*\.txt', categ_pattern=r'(.*?)/.*\.txt',
                              encoding=files_encoding)

        file_ids = [os.path.join(path_to_load,item) for item in reader.root_ids]
        for file_id in file_ids:
            results.append((self.start_ngr_calc(files_encoding=files_encoding, file_id=file_id,load_new_top_ngrms=load_new_top_ngrms, filter_ngr_len=filter_ngr_len)))
        return results
    
    def load_new_dir(self, path_to_load, files_encoding,load_new_top_ngrms='',flag_async=True, filter_ngr_len=0):
        if flag_async: results=self.load_new_dir_async(path_to_load, files_encoding,load_new_top_ngrms, filter_ngr_len) 
        else: results=self.load_new_dir_sync(path_to_load, files_encoding,load_new_top_ngrms, filter_ngr_len)
        
        self.logger.info('finished ngr_counting for all {} files '.format(len(results)))
        # учесть то, что было до этого загружено
        results_new = results.copy()
        if os.path.exists(os.path.dirname(self.results_dir_name) + self.ngram_preserver_name):
                results.append((self.ngram_preserver.allgrams,self.ngram_preserver.ngrams,self.ngram_preserver.unigrams))
        
        self.ngram_preserver = self.ngram_preserver_from_uni_dicts(results)
        self.ngram_preserver.save(self.ngram_preserver_name)
        
        if os.path.exists(os.path.join(path_to_load) + ' loaded'):
            shutil.rmtree(os.path.join(path_to_load) + ' loaded')    
        os.rename(os.path.join(path_to_load), os.path.join(path_to_load) + ' loaded')
        
        return results_new
            
    def start_ngr_calc(self, file_id, files_encoding,load_new_top_ngrms='', filter_ngr_len=0):
        path_to_load = os.path.dirname(file_id)
        reader = CorpusReader(input_folder_name=path_to_load,\
                              doc_pattern=r'(.*?/).*\.txt', categ_pattern=r'(.*?)/.*\.txt',
                              encoding=files_encoding)

        ngram_preserver = NgramPreserver(self.ngr_num)
        return ngram_preserver.count_ngrams_file(reader, file_id,load_new_top_ngrms, filter_ngr_len)




class NgramPreserver():

    def __init__(self, n,saved_filename=None):

        self.logger = setup_logger('preserver', '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                   "ngram_logger.log")
        self.n = n

        self.entropy_logger_filename = 'entropy_calc.txt'

        if saved_filename:
            self.load(saved_filename)
        else:
            self.allgrams = defaultdict(ConditionalFreqDist)
            self.ngrams = FreqDist()
            self.unigrams = FreqDist()

        self.dict_len = len(self.unigrams)

    def init_dicts(self, dict_al_gr, dict_n_gr, dict_uni):
        self.allgrams = dict_al_gr
        self.ngrams = dict_n_gr
        self.unigrams = dict_uni

        self.dict_len = len(dict_uni)
	
	
    def count_ngrams_text(self, training_text,load_new_top_ngrms='', filter_ngr_len=0):

        for sent in training_text:
            sent_start = True
            if len(list(ngrams(word_tokenize(sent), self.n)))>0:
                for ngram in ngrams(word_tokenize(sent), self.n):
                    self.ngrams[ngram] += 1
                    context, word = tuple(ngram[:-1]), ngram[-1]
                    if sent_start:
                        for context_word in context:
                            self.unigrams[context_word] += 1
                        sent_start = False
    
                    for window, ngram_order in enumerate(range(self.n, 1, -1)):
                        context = context[window:]
                        if load_new_top_ngrms and window==0:
                            
                            if context not in load_new_top_ngrms.keys():
                                continue
                            else:
                                filt_l=[len(item)>filter_ngr_len for item in context]                                
                                if np.all(filt_l) and len(word)>filter_ngr_len:
                                    self.allgrams[ngram_order][context][word] += 1                                    

                        else:
                            self.allgrams[ngram_order][context][word] += 1
                    self.unigrams[word] += 1
            else:
                for word in word_tokenize(sent):
                    self.unigrams[word] += 1
                
        self.dict_len = len(self.unigrams)

    def count_ngrams_file(self, reader, file_id,load_new_top_ngrms='', filter_ngr_len=0):
        file_id = os.path.basename(file_id) 
        doc_sents = list(reader.sents(fileids=[file_id]))
        self.count_ngrams_text(doc_sents,load_new_top_ngrms, filter_ngr_len)

        self.logger.info('finished ngr_counting for file - {}'.format(file_id))

        return (self.allgrams,self.ngrams,self.unigrams)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(({'allgrams': self.allgrams, 'ngrams': self.ngrams, 'unigrams': self.unigrams}),f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            dict = pickle.load(f)

            self.allgrams = dict['allgrams']
            self.ngrams = dict['ngrams']
            self.unigrams = dict['unigrams']



def ngram_lexicon_write(ngram_preserver, ngram_rank,filename,encoding_file):
    with open(filename,'wt',encoding=encoding_file) as f_w:
        contexts=ngram_preserver.allgrams[ngram_rank]
        for context in contexts:
            f_w.write(' '.join(context) + '(freq - {}):\n'.format(contexts[context].N()))
            for item in contexts[context]:
                f_w.write(item + '({}),'.format(contexts[context][item]))
            f_w.write('\n')



    
    
if __name__=='__main__':

    ngrc = NgrammConsolidator('settings.txt')
        



