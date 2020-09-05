
import multiprocessing as mp
import os
import shutil
import sys
sys.path.append('../')
import general_modules.text_analysis
from general_modules.reader import CorpusReader
from general_modules.multi_magic import MultiFilesOperator
import pickle
import pandas as pd


from general_modules.util import setup_logger

# fmt='%(asctime)s %(levelname)s %(message)s'
# logging.basicConfig(level='INFO', filename='stemmer_logger.log',format=fmt)
# logger = logging.getLogger('stemmer_log')


RESULTS_DIR_NAME='results'
STEM_DIR_NAME = 'forStem'
TEMP_DIR_NAME = 'temp'


class StemmerFiles():

    saved_files_to_stem_path='saved_files_to_stem_path'


    def __init__(self, input_path,file_paths,encoding='utf-8', process_num=mp.cpu_count(), \
            nltk_stop_lang = 'russian', nltk_stemmer_lang = 'russian', stop_words_files = ['словари/delete.txt','словари/delete_sym.txt'], \
            dict_stem_file = 'словари/морфологический словарь.txt', min_token_len=3, \
            results_dir_name=RESULTS_DIR_NAME,stem_dir_name = STEM_DIR_NAME, temp_dir_name = TEMP_DIR_NAME, save_sents=False):

        self.logger = setup_logger('stemmer_log', '%(asctime)s - %(levelname)s - %(message)s',
                                   "stemmer_logger.log")

        self.stem_dir_name = stem_dir_name
        self.temp_dir_name = temp_dir_name
        self.results_dir_name=results_dir_name
        self.stop_words_files = stop_words_files
        self.dict_stem_file = dict_stem_file
        self.nltk_stemmer_lang = nltk_stemmer_lang
        self.save_sents=save_sents
        dict_stem = pd.read_csv(dict_stem_file, sep='|', encoding='cp1251')
        dict_stem = dict_stem.dropna()
        dict_stem = dict_stem.set_index('стем')
        self.dict_stem_file = dict_stem
        
        self.nltk_stop_lang = nltk_stop_lang
        #self.input_path = input_path
        self.output_dirname = os.path.basename(input_path) if not os.path.dirname(file_paths[0]) else os.path.dirname(file_paths[0])
        self.min_token_len =min_token_len
        self.encoding=encoding
        self.process_num=process_num
        results_path=os.path.join(os.getcwd(),results_dir_name)
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        self.stem_dir_name = os.path.join(results_dir_name,self.stem_dir_name)
        if not os.path.exists(self.stem_dir_name):
            os.mkdir(self.stem_dir_name)
        self.stem_temp_dir_name = os.path.join(self.stem_dir_name,self.temp_dir_name)
        if not os.path.exists(self.stem_temp_dir_name):
            os.mkdir(self.stem_temp_dir_name)

        #file_paths_for_stem содержит кортежи какой файл стемить и куда записывать
        if os.path.exists(self.saved_files_to_stem_path):
            with open(self.saved_files_to_stem_path, 'rb') as f:
                    self.file_paths_for_stem=pickle.load(f)

        else: self.file_paths_for_stem=[]
        if self.really_new_files(file_paths):
            for file_name in file_paths:
                shutil.copy(input_path+'\\'+file_name, self.stem_temp_dir_name)
                # этот словарь не используется, подразумевался для запоминания недостемленных
                self.file_paths_for_stem.append((file_name,self.get_new_path(file_name)))
        self.logger.info('in directory - {} {} files to stem'.format(self.temp_dir_name,len(self.file_paths_for_stem)))

    def get_new_path(self, file_name):

        return os.path.join(os.getcwd(),self.results_dir_name,self.output_dirname,os.path.basename(file_name))
    
    def on_result(self,result):
        self.logger.info('finished and deleted file - {}'.format(result))
        os.remove(result)

    def really_new_files(self,file_paths):
        set_old=set(os.path.basename(item) for item in file_paths)
        set_new = set(os.path.basename(item[1]) for item in self.file_paths_for_stem)
        if not set_new:
            return True
        set_new = set_new.union(set_old)

        return not len(set_old)==len(set_new)


    def start_stem_files_async(self):
        mfo = MultiFilesOperator(process_num=self.process_num)
        mfo.multi_files_operator(func_oper=self.stem_file,
                                           path_to_load=self.stem_temp_dir_name, 
                                           files_encoding=self.encoding,callback=(self.on_result,))
            
    def start_stem_files_sync(self):        
        reader = CorpusReader(input_folder_name=self.stem_temp_dir_name,\
                              doc_pattern=r'(.*?/).*\.txt', categ_pattern=r'(.*?)/.*\.txt',
                              encoding=self.encoding)
        file_ids = [os.path.join(self.stem_dir_name+'\\'+self.temp_dir_name,item) for item in reader.root_ids]
        for file in file_ids:
            self.on_result(self.stem_file(file, self.encoding))
                
    
    def start_stem_files(self,flag_async=True):
        try:
            self.start_stem_files_async() if flag_async else self.start_stem_files_sync()
            
            self.logger.info('all operations were completed')
            os.rmdir(self.stem_temp_dir_name)
            os.rmdir(self.stem_dir_name)
            if os.path.exists(self.saved_files_to_stem_path):
               os.remove(self.saved_files_to_stem_path)
               
        except Exception as e:            
            self.logger.info('exception saving files_to_stem {}'.format(e))
            with open(self.saved_files_to_stem_path,'wb') as f:
                pickle.dump(self.file_paths_for_stem,f)

        
    def stem_file(self,file_name, file_encoding):
        file_name_out=self.get_new_path(file_name)
        write_lines=[]
        with open(file_name, 'rt',encoding=file_encoding) as f_in:
            for line in f_in:
                write_lines.append(general_modules.text_analysis.preprocess_text(line, 
                                                      self.nltk_stop_lang, 
                                                      self.nltk_stemmer_lang,
                                                      self.min_token_len,
                                                      self.stop_words_files,
                                                      self.dict_stem_file,save_sents=self.save_sents) +'\n')
        if not os.path.exists(os.path.dirname(file_name_out)):
            os.mkdir(os.path.dirname(file_name_out))

        with open (file_name_out,'wt',encoding=file_encoding) as f_out:
            f_out.writelines(write_lines)

        return file_name
    


if __name__=='__main__':

    INPUT_FOLDER='input_texts/temp3'
    #INPUT_FOLDER='input_texts'
    #INPUT_FOLDER = 'input_texts2'
    
    reader = CorpusReader(input_folder_name=INPUT_FOLDER, doc_pattern=r'(.*?/).*\.txt', categ_pattern=r'(.*?)/.*\.txt',
                          encoding='utf8')

    fileids = reader.root_ids
    #fileids = reader._resolve(categories=['temp3'])
    stef=StemmerFiles(INPUT_FOLDER,fileids,encoding='utf8')
    
    stef.stem_file('input_texts/temp3/export 2019-12-16 1175.txt', 'utf-8')
    
    # stef.start_stem_files(flag_async=False)


