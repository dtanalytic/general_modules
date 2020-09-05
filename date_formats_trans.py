from datetime import datetime
import time

def epoch2structtime(milisec,local=False):
    return time.localtime(milisec) if local else time.gmtime(milisec)

def structtime2datetime(structtime):
    datetime_obj=datetime(structtime.tm_year,structtime.tm_mon, structtime.tm_mday,structtime.tm_hour,structtime.tm_min,structtime.tm_sec)
    return datetime_obj

def epoch2datetime(milisec,local=False):
    structtime=epoch2structtime(milisec,local)
    return structtime2datetime(structtime)

def structtime2epoch(structtime):
    return time.mktime(structtime)

def datetime2structtime(datetime_obj):
    datetime_str='{}-{}-{} {}:{}:{}'.format(datetime_obj.year,datetime_obj.month,datetime_obj.day,datetime_obj.hour,datetime_obj.minute,datetime_obj.second)
    return time.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")

def datetime2epoch(datetime_obj):
    strtime=datetime2structtime(datetime_obj)
    return structtime2epoch(strtime)

if __name__=='__main__':

    pass

