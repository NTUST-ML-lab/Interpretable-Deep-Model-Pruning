import sys
import os

class SaveOutput():
    def __init__(self, logfile):
        self.stdout = sys.stdout
        toks = logfile.split("/")[:-1]
        strr = ""
        for tok in toks:
            strr += tok + "/"
        checkkk(strr[:-1])
        self.log = open(logfile, 'w')
        
    def write(self, text):
        self.stdout.write(text)
        self.log.write(text)
        self.stdout.flush()
        self.log.flush()
            
    def close(self):
        self.stdout.close()
        self.log.close()
        
    def flush(self):
        pass

def checkkk(strr):
    '''
    check path available, if not, it will create.
    return absolute path of input (end withou / )
    '''
    strr = strr.replace("\n", "")
    path = os.path.split(os.path.realpath(__file__))[0] + "/" + strr
    if not os.path.isdir(path):
        os.makedirs(path)
    return path    