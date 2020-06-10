import torch
import torch.nn as nn
import numpy as np

class NetworkFit(object):
    def __init__(self, model, optimizer, soft_criterion):
        self.model = model
        self.optimizer = optimizer
        
        self.soft_criterion = soft_criterion
        
    
    def train(self, inputs, labels):
        self.optimizer.zero_grad()
        self.model.train()

        outputs = self.model(inputs)
            
        soft_loss = self.soft_criterion(outputs, labels)
        loss = soft_loss
        
        loss.backward()
        self.optimizer.step()
            
            
    def test(self, inputs, labels):
        self.model.eval()
        
        outputs = self.model(inputs)
        
        soft_loss = self.soft_criterion(outputs, labels)
        loss = soft_loss
        
        _, predicted = outputs.max(1)
        correct = (predicted == labels).sum().item()
        
        return [loss.item()], [correct]
        
    def get_model(self):
        return self.model