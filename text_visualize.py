import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from prettytable import PrettyTable
from yellowbrick.text.freqdist import FreqDistVisualizer
from yellowbrick.text.freqdist import FrequencyVisualizer
from general_modules.reader import CorpusReader

class TextVisualizeTokens():

    
    @classmethod
    def draw_word_cloud(cls,freq_dict, title):
        fig, ax = plt.subplots(1, 1)
        wordcloud=WordCloud(relative_scaling=0.5).generate_from_frequencies(freq_dict)
        ax.set_title('частота слов {}'.format(title))
        ax.imshow(wordcloud)
        ax.axis('off')
    
    @classmethod
    def word_cloud_file(cls,freq_dict,out_file_path,max_words_in_file):

        wordcloud=WordCloud(relative_scaling=0.5,max_words=max_words_in_file).generate_from_frequencies(freq_dict)
    
        wordcloud.to_file(out_file_path)

    @classmethod
    def hist_significant_tokens_file(cls, word_counter_dict, output_file_path, max_words_in_file=30):
            # если обычный словарь,не Counter    
            if not isinstance(word_counter_dict,Counter):
                word_counter_dict = Counter(word_counter_dict)
            
            most_common_list = word_counter_dict.most_common(max_words_in_file)
            most_common_words = [item[0] for item in most_common_list]
            most_common_stats = [item[1] for item in most_common_list]
    
            xs = [i + 0.1 for i, _ in enumerate(most_common_words)]
            plt.bar(xs, most_common_stats)
            plt.xticks([i + 0.5 for i, _ in enumerate(most_common_words)], most_common_words, rotation=90)
    
            plt.xlabel('слова')
            plt.ylabel('частота')
            plt.title('частота наиболее значимых слов')
            plt.savefig(output_file_path)

    # если малая признаковая база может не сработать
    @classmethod
    def hist_tokens_texts(cls, texts_dir,vectorizer, ext='txt'):
        reader = CorpusReader(input_folder_name=texts_dir, doc_pattern=r'(.*?/).*\.'+ext, 
                              categ_pattern=r'(.*?)/.*\.'+ext,
                              encoding='utf-8')
        texts = list(reader.readfiles(fileids=reader.root_ids))
    
        docs = vectorizer.fit_transform(texts)
        
        features   = vectorizer.get_feature_names()
        
        visualizer = FreqDistVisualizer(
            features=features, size=(1080, 720)
        )
        visualizer.fit(docs)
        visualizer.show()
        #visualizer.poof()

        
    @classmethod
    def list2pretty_table(cls,vis_list, columns, filename):
        pt = PrettyTable(field_names=columns)
        for item in vis_list:
            pt.add_row([str(col) for col in item])
        table_txt = pt.get_string()
        with open(filename,'w') as file:
            file.write(table_txt)   
            

if __name__=='__main__':
    
    
    import sys
    sys.path.append('../')
    from general_modules.text_analysis import StemmedCountVectorizer
    from nltk.stem import SnowballStemmer
    from collections import Counter
    from general_modules.text_stats import words_counts
    
    vectorizer = StemmedCountVectorizer(SnowballStemmer('russian'), min_len_token=4)
    
    TextVisualizeTokens().hist_tokens_texts('input_texts/temp3',vectorizer, ext='txt')
    string1 = 'жили Были, S квартире Сирии, Сирия 1912 два кота и дочка\nкрасивая-такая\nкоторого'

        
    
    #df=get_most_common_words('электроцинкубивает.json', 'utf-8', vectorizer)
    

    freq_dict = words_counts(string1,vectorizer)     

    TextVisualizeTokens.word_cloud_file(freq_dict,'cloud.png',10)

    TextVisualizeTokens.hist_significant_tokens_file(freq_dict, 'hist.png')
    TextVisualizeTokens.draw_word_cloud(freq_dict,'график')