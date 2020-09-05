from nltk import FreqDist
from string import punctuation
from nltk.corpus.reader.plaintext import CategorizedPlaintextCorpusReader
from nltk import sent_tokenize
from nltk import ngrams
from nltk.tokenize import word_tokenize
import os
from nltk.corpus.reader.plaintext import PlaintextCorpusReader


class CorpusReader(CategorizedPlaintextCorpusReader):

    def __init__(self, input_folder_name,doc_pattern,categ_pattern,encoding='utf-8'):
        CategorizedPlaintextCorpusReader.__init__(self, input_folder_name, doc_pattern, cat_pattern=categ_pattern)
        self.input_folder_name = input_folder_name
        self.encoding = encoding
        self.root_reader=PlaintextCorpusReader(input_folder_name,fileids=r'[^\/]*.'+doc_pattern[-3:])
        #self.root_ids =[ os.path.join(input_folder_name,item) for item in self.root_reader.fileids()]
    
        self.root_ids = list(self.root_reader.fileids())
    
    def fileids(self, categories=None):
        ids=super().fileids(categories=categories)
        if ids: return ids    
        else: return self.root_ids
        
        
    def _resolve(self, fileids=None, categories=None):

        if fileids is not None and categories is not None:
            raise ValueError("Необходимо задать что-то одно")

        if categories is not None:
            return self.fileids(categories)
        if fileids is not None:
            return fileids

        return self.fileids()

    
    def paras(self, delim='\n', fileids=None, categories=None):
        for file_text in self.readfiles(fileids, categories):
            for paragraph in file_text.split(delim):
                yield paragraph

    def sents(self, delim='\n', fileids=None, categories=None):
        for para in self.paras(delim, fileids, categories):
            for sent in sent_tokenize(para):
                yield sent

    def words(self, delim='\n', fileids=None, categories=None, ignore_digits=False, ignore_punct=False):
        if ignore_punct:
            punct=list(punctuation)
            my_punct = ['\'','«', '»', '``', '\"',"''"]
            punct.extend(my_punct)
        for sent in self.sents(delim, fileids, categories):
            for word in word_tokenize(sent):
                if ignore_punct and ignore_digits:
                    if not word.isdigit() and not word in punct:
                        yield word 
                elif ignore_punct:
                    if not word in punct:
                        yield word
                elif ignore_digits:
                    if not word.isdigit():  
                        yield word 
                else:
                    yield word 
                

    def bag_words(self, paras_delim='\n', fileids=None, categories=None):
        for file_text in self.readfiles(fileids, categories):
           yield [w for paragr in file_text.split(paras_delim) for w in word_tokenize(paragr)]

    def bag_words_tokenized(self, paras_delim='\n', fileids=None, categories=None):
        for file_text in self.readfiles(fileids, categories):
           yield [w for paragr in file_text.split(paras_delim) for w in paragr.split()]

    def ps_s_w_files(self, paras_delim='\n', fileids=None, categories=None):
        for file_text in self.readfiles(fileids, categories):
            yield [[word_tokenize(sent)\
                for sent in sent_tokenize(par)] \
                for par in file_text.split(paras_delim)]

    def readfiles(self, fileids=None, categories=None):
        fileids = self._resolve(fileids, categories)
        for filename in fileids:
            with open('{}\{}'.format(self.root, filename),encoding=self.encoding) as f:
                yield f.read()

    def ngram_files(self, ngr_len, paras_delim='\n', fileids=None, categories=None):
        for sent in self.sents(paras_delim, fileids, categories):
            for ngram in ngrams(word_tokenize(sent),ngr_len):
                yield ngram
    

if __name__=='__main__':

    from string import punctuation
    reader = CorpusReader(input_folder_name='input_texts', doc_pattern=r'(.*?/).*\.txt', categ_pattern=r'(.*?)/.*\.txt',encoding='utf-8')
 
    
    fileids=reader.root_ids
    text = list(reader.readfiles(fileids=reader.root_ids))
    
    ngrs = list(reader.ngram_files(ngr_len=3,fileids=reader.root_ids))  
    wrds = list(reader.words(fileids=reader.root_ids,ignore_punct=True)) 
    sents= list(reader.sents(fileids=reader.root_ids))
    
    ids=reader.fileids()
    
    print('wrds_num = ngr_num + sents_num: {} = {} + {}'.format(len(wrds),len(ngrs),len(sents)))
    # sents = list(reader.sents(fileids=['temp2/export 2019-12-16 1183.txt']))
    # list2 = word_tokenize(sents[1])
    # bags2 = list(reader.ngram_files(3, fileids=['temp2/export 2019-12-16 1183.txt']))
    #
    # punct=list(punctuation)
    # list_words = [w for w in list2 if not w.isdigit() and not w in punct]



