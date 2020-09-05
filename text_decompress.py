from gensim.summarization import summarize
import nltk
import numpy as np
import sys
sys.path.append('../')
from general_modules.text_stats import words_counts
from general_modules.text_visualize import TextVisualizeTokens


# from IPython.display import IFrame
# from IPython.core.display import display

# HTML_TEMPLATE = """<html>
#     <head>
#         <title>{0}</title>
#         <meta http-equiv="Content-Type" content="text/html; charset=UTF-8"/>
#     </head>
#     <body>{1}</body>
# </html>"""


sys.path.append('../')
from general_modules.text_clean import HtmlTextCleaner

def score_sentences(sentences, important_words, cluster_threshold):
        scores = []
        sentence_idx = -1
    
        for s in [nltk.tokenize.word_tokenize(s) for s in sentences]:
    
            sentence_idx += 1
            word_idx = []
    
            for w in important_words:
                try:    
                    word_idx.append(s.index(w))
                except ValueError as e: 
                    print(e)
    
            word_idx.sort()
    
            if len(word_idx)== 0: continue
    
            clusters = []
            cluster = [word_idx[0]]
            i = 1
            while i < len(word_idx):
                if word_idx[i] - word_idx[i - 1] < cluster_threshold:
                    cluster.append(word_idx[i])
                else:
                    clusters.append(cluster[:])
                    cluster = [word_idx[i]]
                i += 1
            clusters.append(cluster)

            max_cluster_score = 0
            for c in clusters:
                significant_words_in_cluster = len(c)
                total_words_in_cluster = c[-1] - c[0] + 1
                score = 1.0 * significant_words_in_cluster \
                    * significant_words_in_cluster / total_words_in_cluster
    
                if score > max_cluster_score:
                    max_cluster_score = score
    
            scores.append((sentence_idx, score))
    
        return scores   

class TextDecompressor():
    
        
    @classmethod
    def decompress_lune(cls,text_in, n=100, cluster_threshold=5, ratio=0.1, top_sentences=30):

        sentences  = nltk.tokenize.sent_tokenize(text_in)
        
        normalized_sentences = [s.lower() for s in sentences]
                
        words = [w.lower() for sentence in normalized_sentences for w in
                 nltk.tokenize.word_tokenize(sentence)]
    
        fdist = nltk.FreqDist(words)
    
        top_n_words = [w[0] for w in fdist.items() 
                if w[0] not in nltk.corpus.stopwords.words('russian')][:n]
    
        scored_sentences = score_sentences(normalized_sentences, top_n_words, cluster_threshold)

        # 1 вариант score > avg + 0.5 * std]
        avg = np.mean([s[1] for s in scored_sentences])
        std = np.std([s[1] for s in scored_sentences])
        mean_scored = [(sent_idx, score) for (sent_idx, score) in scored_sentences
                       if score > avg + 0.5 * std]
    
        # 2 вариант n самых важных
         
        top_n_scored = sorted(scored_sentences, key=lambda s: s[1])[-top_sentences:]
        top_n_scored = sorted(top_n_scored, key=lambda s: s[0])
    
        top_ratio = sorted(scored_sentences, key=lambda s: s[1])[-np.int(len(sentences)*ratio):]
        top_ratio = sorted(top_ratio, key=lambda s: s[0])
        
    
        return dict(top_n_summary=[sentences[idx] for (idx, score) in top_n_scored],
                    mean_scored_summary=[sentences[idx] for (idx, score) in mean_scored],
                    top_ratio_summary= [sentences[idx] for (idx, score) in top_ratio])
    
    @classmethod
    def visualize(cls, text_sents, main_sents,filename_out):
        HTML_TEMPLATE = """<html>
            <head>
                <title>{0}</title>
                <meta http-equiv="Content-Type" content="text/html; charset=UTF-8"/>
            </head>
            <body>{1}</body>
        </html>"""
        
        new_sents = '<p>{0}</p>'.format(text_sents)
        
        for s in main_sents:
                new_sents = new_sents.replace(s, '<strong>{0}</strong>'.format(s))
            
        with open(filename_out, 'wb') as f:                
            html = HTML_TEMPLATE.format('Summary',new_sents)    
            f.write(html.encode('utf-8'))
    
        print("Data written to", filename_out)
        
        
    
if __name__=='__main__':
    
    from nltk.stem import SnowballStemmer
    from collections import Counter
    
    import sys
    sys.path.append('../')    
    from general_modules.text_analysis import StemmedCountVectorizer
    from general_modules.text_stats import words_counts
    from general_modules.text_stats import crop_freq_dict    
    from general_modules.util import save_struct,load_struct
    
    #..................................................................
    # стадия загрузки  и фильтрации текста
    # with open('бпк1.html','rt', encoding='utf-8') as f:
    #     text=f.read()
    # t = HtmlTextCleaner().boilerpipe_text(html_in=text,extractor='KeepEverythingExtractor')       
    # #t2 = HtmlTextCleaner().bs_text(html_in=text)
       
    
    # t = HtmlTextCleaner.clean_part(t, txt_phraze=('Я продолжаю эту традицию.',
    #                                               'Да, спасибо, хорошо'), where='between')
    # t = HtmlTextCleaner.clean_part(t, txt_phraze=('позавчерашнего времени. Итак.',
                                                  # 'Я вас всех поздравляю с наступающим Новым годом'), where='between')
    # with open('pr2.txt' , 'wt', encoding='utf-8') as f:
    #     f.write(t)
        
        
    # #..................................................................
    # # подсчет частот при наличии текста
    # vectorizer = StemmedCountVectorizer(SnowballStemmer('russian'), min_len_token=3, 
    #                                     files_delete=['словари/delete.txt',('словари/spec.txt',0),'словари/delete_sym.txt', ('словари/stops.txt',1)]
    #                                     )
    # freq_dict = words_counts(t,vectorizer)
       
    # freq_dict_old = freq_dict.copy()
    # save_struct('2018_press_words',freq_dict_old)
    
    # ....................................................................
    # визуализация
    # TextVisualizeTokens().word_cloud_file(freq_dict,'cloud.png',10)
    # TextVisualizeTokens().hist_significant_tokens_file(freq_dict, 'hist.png')
    # TextVisualizeTokens().draw_word_cloud(freq_dict,'график')
    
    
    #....................................................................
    # работа с частотными словарями
    
    freq_dict_old1 = dict(load_struct('2018_press_words'))
    freq_dict_old2 = dict(load_struct('2019_press_words'))
    
    # pr_w1 = Counter(freq_dict_old1).most_common()
    # pr_w2 = Counter(freq_dict_old2).most_common()
    pr_w1, borders1 = crop_freq_dict(freq_dict_old1, 0.1,0.95)
    pr_w2, borders2 = crop_freq_dict(freq_dict_old2, 0.1,0.95)

    pr_w1 = pr_w1.most_common()
    pr_w2 = pr_w2.most_common()
    
    s1 = set([item[0] for item in pr_w1])
    s2 = set([item[0] for item in pr_w2])

    
    gen_w = sorted([(item,freq_dict_old1[item],freq_dict_old2[item]) for item in s1.intersection(s2)],key=lambda x: x[1],reverse=True)
    s1_un = sorted([(item,freq_dict_old1[item]) for item in s1.difference(s2)],key= lambda x: x[1],reverse=True)
    s2_un = sorted([(item,freq_dict_old2[item]) for item in s2.difference(s1)], key= lambda x:x[1], reverse=True)
    
    TextVisualizeTokens.list2pretty_table(gen_w,columns=['слово', 'частота 1', 'частота 2'],filename = 'gen_w.txt' )
    TextVisualizeTokens.list2pretty_table(s1_un,columns=['слово', 'частота'],filename = 's1_un.txt' )
    TextVisualizeTokens.list2pretty_table(s2_un,columns=['слово', 'частота'],filename = 's2_un.txt' )

    #
    # freq_dict_old1['бензин']
    
    # freq_dict_old2['бензин']
    

    #....................................................................
    # аннотация
    # dict_sum=TextDecompressor().decompress_lune(text_in=t, n=100, \
    #                                             cluster_threshold=5, top_sentences=5)
    # dict_sum['gensim'] = summarize(t,ratio=0.01,split=True)
    
    

    
    # for summary_type in ['top_n_summary', 'mean_scored_summary', 'top_ratio_summary', 'gensim']:
    #     TextDecompressor().visualize(t, dict_sum[summary_type],summary_type+'.html')
        
    #     for s in dict_sum[summary_type]:
    #             dict_sum[summary_type + '_marked_up'] = \
    #             dict_sum[summary_type + '_marked_up'].replace(s, '<strong>{0}</strong>'.format(s))
    
    #     filename = 'summary.' + summary_type + '.html'
    #     f = open(filename, 'wb')
    #     html = HTML_TEMPLATE.format('Summary', dict_sum[summary_type + '_marked_up'])    
    #     f.write(html.encode('utf-8'))
    #     f.close()
    
    #     print("Data written to", f.name)
    
    
    
