class ILogger:
    def println(self, print_str: str):
        pass
    
class ConsoleLogger(ILogger):
    def println(self, print_str: str):
        print(print_str)
        
        
class FileLogger(ILogger):
    def __init__(self):
        super().__init__()
        
    def open(self, all_path: str, option: str):
        self.file = open(all_path, option, encoding="utf-8")
    
    def close(self):
        if self.file:
            self.file.close()
        
    def println(self, print_str: str):
        self.file.write(print_str + "\n")
        
class SaveLogger(ILogger):
    def __init__(self) -> None:
        super().__init__()
        self.log_list = []
        
    def println(self, print_str: str):
        self.log_list.append(print_str)
        
    def get_str(self):
        final_log = ""
        for log in self.log_list:
            final_log += log
        
        return final_log
    
    def clear_log(self):
        self.log_list.clear()