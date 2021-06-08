import random
import multiprocessing
from joblib import Parallel, delayed
from copy import deepcopy

from .model import *
from .constants import *
from .gosh_acq import *
from adahessian import Adahessian

num_cores = multiprocessing.cpu_count()

class GOSH():
	def __init__(self, input_dim, bounds, implement_gobi, trust_region, second_order, parallel, model_aleatoric, pretrained):
		assert bounds[0].shape == input_dim and bounds[1].shape == input_dim
		self.input_dim = input_dim
		self.bounds = bounds
		self.trust_region = trust_region
		self.second_order = second_order
		self.run_aleatoric = model_aleatoric
		self.init_models(pretrained)

	def init_models(self, pretrained):
		self.student = student(self.input_dim)
		self.teacher = teacher(self.input_dim)
		self.student_opt = torch.optim.AdamW(self.student.parameters() , lr=LR)
		self.teacher_opt = torch.optim.AdamW(self.teacher.parameters() , lr=LR)
		self.student_l, self.teacher_l = [], []
		self.epoch = -1
		if self.run_aleatoric:
			self.npn = npn(self.input_dim)
			self.npn_opt = torch.optim.AdamW(self.npn.parameters() , lr=LR)
			self.npn_l = []
		if pretrained:
			self.student, self.student_opt, _, self.student_l = load_model(self.student, self.student_opt)
			self.teacher, self.teacher_opt, self.epoch, self.teacher_l = load_model(self.teacher, self.teacher_opt)
			if self.run_aleatoric:
				self.npn, self.npn_opt, self.epoch, self.npn_l = load_model(self.npn, self.npn_opt)

	def train(self, xtrain, ytrain):
		self.epoch += 1
		teacher_loss = self.train_teacher(xtrain, ytrain)
		student_loss = self.train_student(xtrain, ytrain)
		npn_loss = self.train_npn(xtrain, ytrain) if self.model_aleatoric else 0
		return npn_loss, teacher_loss, student_loss

	def predict(self, x):
		'''
		input: x - list of queries
		outputs: (predictions, uncertainties) where uncertainties is a pair of al and ep uncertainties
		'''
		if not self.run_aleatoric: self.teacher.eval()
		with torch.no_grad():
			outputs = []
			for feat in x:
				feat = torch.tensor(feat, dtype=torch.float)
				if self.run_aleatoric:
					pred, al = self.npn(feat)
				else:
					pred, al = self.teacher(feat), 0
				ep = self.student(feat)
				outputs.append((pred, (ep, al)))
		if not self.run_aleatoric: self.teacher.train()
		return outputs

	def get_queries(self, x, k, explore_type):
		'''
		x = list of inputs
		k = integer (batch of queries i.e. index of x closest my opt output)
		explore_type (str, optional): type of exploration; one
            in ['ucb', 'ei', 'pi', 'ts', 'percentile', 'mean', 
            'confidence', 'its', 'unc']. 'unc' added for purely
            uncertainty-based sampling 
		'''
		threads = max(num_cores, k) if self.parallel else 1
		inits = random.choices(x, k=k)
		if not self.run_aleatoric: self.teacher.eval()
		self.freeze()
		inits = Parallel(n_jobs=threads, backend='threading')(delayed(self.parallelizedFunc)(i, explore_type) for i in inits)
		self.unfreeze()
		indices = []
		for init in inits:
			devs = torch.mean(torch.abs(init - torch.from_numpy(np.array(x))))
			indices.append(torch.argmin(devs).item())
		if not self.run_aleatoric: self.teacher.train()
		return indices

	def parallelizedFunc(self, init, explore_type):
		init = torch.tensor(init, dtype=torch.float, requires_grad=True)
	    optimizer = torch.optim.AdamW([init] , lr=0.8) if not self.second_order else Adahessian([init] , lr=0.8)
	    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
	    iteration = 0; equal = 0; z_old = 100; zs = []
	    while iteration < 200:
	        old = deepcopy(init.data)
	        if self.trust_region:
	        	trust_bounds = (old*(1-trust_region), old*(1+trust_region))
			pred = self.npn(init)[0] if self.run_aleatoric else self.teacher(init)
	        ep = self.student(init)
	        z = gosh_acq(pred, ep)
	        optimizer.zero_grad(); z.backward(); optimizer.step(); scheduler.step()
	        init = torch.max(self.bounds[0], torch.min(self.bounds[1], init))
	        if self.trust_region:
	        	init = torch.max(trust_bounds[0], torch.min(trust_bounds[1], init))
	        equal = equal + 1 if torch.all((init - old) < epsilon) else 0
	        if equal > 5: break
	        iteration += 1; z_old = z.item()
	    init.requires_grad = False 
	    return init.data

	def freeze(self):
		freeze_models([self.student, self.teacher])
		if self.run_aleatoric: freeze_models([self.npn])

	def unfreeze(self):
		unfreeze_models([self.student, self.teacher])
		if self.run_aleatoric: unfreeze_models([self.npn])

	def train_teacher(self, xtrain, ytrain)
		total = 0
		for feat, y_true in zip(xtrain, ytrain):
			feat = torch.tensor(feat, dtype=torch.float)
			y_pred = self.teacher(feat)
			self.teacher_opt.zero_grad()
			loss = RMSE(y_pred, y_true)
			loss.backward()
			self.teacher_opt.step()
			total += loss
		self.teacher_l.append(total /  len(xtrain))
		save_model(self.teacher, self.teacher_opt, self.epoch, self.teacher_l)
		return self.teacher_l[-1]

	def train_student(self, xtrain, ytrain):
		total = 0
		for feat, y_true in zip(xtrain, ytrain):
			feat = torch.tensor(feat, dtype=torch.float)
			outputs = [self.teacher(feat) for _ in range(Teacher_student_cycles)]
			y_true = torch.std(torch.stack(outputs))
			y_pred = self.student(feat)
			self.student_opt.zero_grad()
			loss = RMSE(y_pred, y_true)
			loss.backward()
			self.student_opt.step()
			total += loss
		self.student_l.append(total /  len(xtrain))
		save_model(self.student, self.student_opt, self.epoch, self.student_l)
		return self.student_l[-1]

	def train_npn(self, xtrain, ytrain):
		total = 0
		for feat, y_true in zip(xtrain, ytrain):
			feat = torch.tensor(feat, dtype=torch.float)
			y_pred = self.npn(feat)
			self.npn_opt.zero_grad()
			loss = Aleatoric_Loss(y_pred, y_true)
			loss.backward()
			self.npn_opt.step()
			total += loss
		self.npn_l.append(total /  len(xtrain))
		save_model(self.npn, self.npn_opt, self.epoch, self.npn_l)
		return self.npn_l[-1]