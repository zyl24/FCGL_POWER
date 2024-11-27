import torch
import torch.nn as nn


class BaseClient:

    def __init__(self, args, client_id, data, message_pool, device):
        self.args = args
        self.client_id = client_id
        self.message_pool = message_pool
        self.device = device
        self.data = data
    
    def execute(self):
        raise NotImplementedError

    def send_message(self):
        raise NotImplementedError


    def task_start(self, task_id):
        pass
    
    def task_done(self, task_id):
        pass
    
class BaseServer:

    def __init__(self, args, message_pool, device):
        self.args = args
        self.message_pool = message_pool
        self.device = device
   
    def execute(self):
        raise NotImplementedError

    def send_message(self):
        raise NotImplementedError
    
    def task_start(self, task_id):
        pass
    
    def task_done(self, task_id):
        pass