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
        self.option = 0
        self.counter = 10
        self.lock = threading.Lock()

    def __shutdown_counter(self):
        """
            Decrementing Counter
        """
        while self.counter >= 0:
            time.sleep(1)
            with self.lock:
                self.counter -= 1
        if self.option == 0:
            sys.stdin.close()
        sys.stdout.write('')
        
    def __get_answer(self):
        """
            Thread handles user's attempt to answer prompt     
        """
        ret = input()
        
        while ret not in ['1','2']:
            sys.stdout.write('Enter Option:')
            ret = input()
        
        if ret.strip() == '1':
            sys.stdout.write('continuing...\n')
        else:
            sys.stdout.write('training ended by user ... \n')
        
        self.option = ret

        with self.lock:
            self.counter = 0

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
    

