import concurrent
import multiprocessing as mp
import sys
sys.path.append('../')
from general_modules.reader import CorpusReader
import os    


class MultiFilesOperator():
    
    def __init__(self,process_num=None):
        self.process_num=process_num if process_num else (mp.cpu_count()-1)


    def multi_files_operator(self, func_oper,path_to_load, files_encoding,**kwargs):
        futures = set()
        call_back_args = None
        reader = CorpusReader(input_folder_name=path_to_load,\
                              doc_pattern=r'(.*?/).*\.txt', categ_pattern=r'(.*?)/.*\.txt',
                              encoding=files_encoding)

        file_ids = [os.path.join(path_to_load,item) for item in reader.root_ids]
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.process_num) as executor:
            if 'callback' in kwargs:
                call_back_args = kwargs['callback']
                del kwargs['callback']
            for file in file_ids:
                future = executor.submit(func_oper,file, files_encoding,**kwargs)
                futures.add(future)
                
            
            results = self.wait_for(futures,call_back_args)

                
                
                
        return results

    def wait_for(self,futures,call_back_args=None):
        results = []
        try:
            for future in concurrent.futures.as_completed(futures):
                err = future.exception()
                if err is None:
                    result = future.result()
                    if call_back_args:
                        call_back_args[0](result,*call_back_args[1:])
                    results.append(result)
        except Exception as e:
            self.logger.info('exception error {}  '.format(e))

        return results

