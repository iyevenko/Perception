import time


class Timer():
    def __init__(self, proc_name, log_file=None):
        self.proc_name = proc_name
        self.log_file = log_file

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        t = time.time() - self.start
        msg = f'{self.proc_name} took {1000*t:.2f} ms'
        if self.log_file is not None:
            with open(self.log_file, 'a') as log:
                log.write(msg + '\n')
        else:
            print(msg)

    @staticmethod
    def clear_logs(log_file):
        open(log_file, 'w').close()
