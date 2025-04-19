import time
import threading 
import sys 

class AsyncReader:
    """reads pipe strings without block main thread
    """
    def __init__(self, stream):
        """
        Constructor.

        Args:
            popen_instance - A stream
        """
        self.stream = stream
        self.data_string = ''
        self.thread = threading.Thread(target=self.__get_lines, daemon=True)
        self.thread.start()
        self.start_time = time.time()
        self.thread2 = threading.Thread(target=self.__get_answer, daemon=True)
        self.thread3 = threading.Thread(target=self.__shutdown_counter)
        self.option = 1
    
    def __shutdown_counter(self):
        time.sleep(10)
        sys.stdout.write('\nTimeout, training will continue\n')
        
    def __get_answer(self):
        """
            Thread handles user's attempt to answer prompt     
        """
        time.sleep(0.5)
        ret = sys.stdin.readline().strip() # read console -1 removes new line at end
        
        while ret not in ['1','2']:
            sys.stdout.write('Enter Option:')
            sys.stdout.flush()
            ret = sys.stdin.readline().strip() # read console
        
        if ret.strip() == '1':
            sys.stdout.write('continuing...\n')
        else:
            sys.stdout.write('training ended by user ... \n')
        
        self.option = ret

    def __get_lines(self):
        """
            Threads continuously reads lines; thread is blocked by stream after reading last line 
        """
        while True:
            self.data_string += self.stream.readline() 

    def get(self):
        self.thread.join(0.1)
        self.thread2.start() 
        self.thread3.start() 
        return self.data_string[:-1]
    

