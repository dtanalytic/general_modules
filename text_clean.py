from boilerpipe.extract import Extractor
from bs4 import BeautifulSoup
from readability.readability import Unparseable
from readability.readability import Document
from file_operations import text2file
import re

class HtmlTextCleaner():
    @classmethod
    def boilerpipe_text(cls,url_in=None,html_in=None,extractor='ArticleExtractor'):
        assert (url_in!=None) != (html_in!=None) # one, not both
        inp=url_in or html_in
        if url_in:
            extractor = Extractor(extractor=extractor, url=inp)
        else:
            extractor = Extractor(extractor=extractor, html=inp)
        return HtmlTextCleaner().spec_text_cleaner(extractor.getText())
    
    @classmethod
    def bs_text(cls,html_in, parser='html.parser'):
        if html_in =='': return ''
        else: 
            soup = BeautifulSoup(html_in,parser)
            text = soup.get_text(" ", strip=True)

            return HtmlTextCleaner().spec_text_cleaner(text)
            
    @classmethod
    def readability_text(cls,html_in):
        try:
            return Document(html_in).summary(html_partial=True)
        except Unparseable as e:
            print('Could not parse HTML: {}'.format(e))

    @classmethod
    def spec_text_cleaner(cls, txt_in):
        text = txt_in.replace('\xa0', ' ')
        text = text.replace('\ufeff', ' ')
        #text = ' '.join(text.split())
        text = re.sub(' +', ' ', text)
        
        return text
    
    @classmethod
    def clean_part(cls, text_in,txt_phraze=None,where='after'):
        '''where : 'after', 'between', 'before'
        Returns: text without some part
        '''
        if where == 'between': assert len(txt_phraze)==2
        if where =='before':            
                text_in = text_in.split(txt_phraze)[1]
        elif where == 'after':             
                text_in = text_in.split(txt_phraze)[0] 
        elif where == 'between':            
                text_in = text_in.split(txt_phraze[0])[1]
                text_in = text_in.split(txt_phraze[1])[0]
                
        return text_in        
                

if __name__=='__main__':
    
    #t1 = HtmlTextCleaner().boilerpipe_text(url_in='https://www.kommersant.ru/doc/4258866?utm_source=yxnews&utm_medium=desktop&utm_referrer=https%3A%2F%2Fyandex.ru%2Fnews')
    
    with open('ziv0_cl.txt','rt', encoding='utf-8') as f:
        text=f.read()
    t2 = HtmlTextCleaner().readability_text(html_in=text)
    
    t3 = HtmlTextCleaner().bs_text(html_in=text)
    t4 = HtmlTextCleaner().boilerpipe_text(html_in=text)
    
    text_in = t4
    # phraze = ('---- 1','... 1965-1970')
    # text_in = HtmlTextCleaner().clean_part(text_in,txt_phraze=phraze,where='between')
    
    # # # # text_in = HtmlTextCleaner().clean_part(text_in,txt_phraze='Испытание',where='before')
    text2file('ziv_cl.txt',text_in)
    
