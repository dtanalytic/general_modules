import os
import shutil
from nltk import sent_tokenize
import json
import pandas as pd
import dateutil.parser


def split_file_pieces(filename,encoding,paras_delim, pieces_dir_name, max_num_paras):
    num_paras=0
    cont_file_piece=''
    cur_num_piece=1
    add_symb=''
    with open(filename,'rt',encoding=encoding) as f:
        if paras_delim:lines=f.readlines()
        else:lines = sent_tokenize(f.read()); add_symb='\n'
        for line in lines:
            cont_file_piece=cont_file_piece+line+add_symb
            if not paras_delim or (paras_delim and line.find(paras_delim)):
                num_paras+=1
                if num_paras>=int(max_num_paras):
                    if not os.path.exists(os.path.join(os.path.dirname(filename),pieces_dir_name)):
                        os.mkdir(os.path.join(os.path.dirname(filename),pieces_dir_name))
                    with open(os.path.join(os.path.dirname(filename),pieces_dir_name,pieces_dir_name+' {} '.format(str(cur_num_piece))+os.path.basename(filename)),'wt',encoding=encoding) as f_piece:
                        f_piece.write(cont_file_piece.lstrip()+'\n')
                        #f_piece.write('aaaaaa')
                        cur_num_piece+=1
                        num_paras=0
                        cont_file_piece=''
        if cont_file_piece !='':
            if not os.path.exists(os.path.join(os.path.dirname(filename),pieces_dir_name)):
                        os.mkdir(os.path.join(os.path.dirname(filename),pieces_dir_name))
                    
            with open(os.path.join(os.path.dirname(filename),pieces_dir_name,pieces_dir_name+' {} '.format(str(cur_num_piece))+os.path.basename(filename)),'wt',encoding=encoding) as f_piece:
                        f_piece.write(cont_file_piece)

def text2file(filename,text_in, encoding='utf8'):
    with open(filename , 'wt', encoding=encoding) as f:
        f.write(text_in)


def split_json_export_file(filename):
    
    json_list=[]
    
    with open(filename,'r',encoding='utf8') as f:
        for line in f:
            json_list.append(json.loads(line))

    df=pd.DataFrame(json_list,columns=['published','message'])

    for i in range(len(df)):
        datetime_obj = dateutil.parser.parse(df.loc[i, 'published'])
        df.loc[i,['message']].to_csv(filename+' '+datetime_obj.strftime('%Y-%m-%d')+' '+str(i)+'.txt',index=False, header=False)





if __name__=='__main__':
    
    split_file_pieces('../text_features/input_texts2/onisrzr_cl.txt','utf8','', 'pieces onisrzr', 10)