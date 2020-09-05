import json
from datetime import datetime
import general_modules.date_formats_trans

def set_trace():
	from IPython.core.debugger import Pdb
	import sys
	Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

def debug(f, *args,**kwargs):
    from IPython.core.debugger import  Pdb
    pdb=Pdb(color_scheme='Linux')
    return pdb.runcall(f, *args,**kwargs)

class ToDictMixin(object):
    def to_dict(self):
        return self._traverse_dict(self.__dict__)

    def _traverse_dict(self, instance_dict):
        output = {}
        for key, value in instance_dict.items():
            output[key] = self._traverse(key, value)
        return output

    def _traverse(self, key, value):
        if isinstance(value, ToDictMixin):
            return value.to_dict()
        if isinstance(value, datetime):
            return general_modules.date_formats_trans.datetime2epoch(value)
        # elif isinstance(value, dict):
        #     return self._traverse_dict(value)
        # elif isinstance(value, list):
        #     return [self._traverse(key, i) for i in value]
        # elif hasattr(value, '__dict__'):
        #     return self._traverse_dict(value.__dict__)
        else:
            return value

    @classmethod
    def from_dict(cls,dict_init):
      obj_ekz = cls()
      #obj_ekz.__dict__= dict_init.copy()
      for key,value in dict_init.items():
          obj_ekz.__dict__[key]=value

      return obj_ekz


class JsonMixin(object):
    @classmethod
    def from_json(cls, file_name):
        with open(file_name + '.json', 'rt') as file:
            kwargs = json.loads(file.read())
            return cls.from_dict(kwargs)

    def to_json(self,file_name):

        with open (file_name+'.json', 'wt') as file:
            file.write(json.dumps(self.to_dict()))


