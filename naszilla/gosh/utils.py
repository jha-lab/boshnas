import os
import pandas as pd 
import numpy as np
import torch
import random

def save_model(model, optimizer, epoch, loss_list):
	file_path = MODEL_SAVE_PATH + "/" + model.name + "_" + str(epoch) + ".ckpt"
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_list': loss_list}, file_path)

def load_model(model, optimizer):
	file_path = MODEL_SAVE_PATH + "/" + model.name + ".ckpt"
	assert os.path.exists(file_path)
	print(color.GREEN+"Loading pre-trained model: "+filename+color.ENDC)
	checkpoint = torch.load(file_path)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	loss_list = checkpoint['loss_list']
	return model, optimizer, epoch, loss_list


def freeze_models(models):
	for model in models:
		for param in model.parameters(): param.requires_grad = False

def unfreeze_models(models):
	for model in models:
		for param in model.parameters(): param.requires_grad = True