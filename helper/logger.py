import time
class TimeLogger:
    def __init__(self):
        self.records = {}  # dictionary to store function names and their max times

    def log_time(self, func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            if func.__name__ not in self.records or self.records[func.__name__] < elapsed_time:
                self.records[func.__name__] = elapsed_time
            return result
        return wrapper
    def add_time_log(self,code_block_name, elapsed_time):
        if code_block_name not in self.records or self.records[code_block_name] < elapsed_time:
            self.records[code_block_name] = elapsed_time
    def print_log(self):
        max_name_length = max(len(name) for name in self.records)
        for name, duration in self.records.items():
            print(f"{name:<{max_name_length + 1}}\t{duration:.6f} sec.")
