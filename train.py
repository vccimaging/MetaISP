''''
Implementation of MetaISP
Code based on the LiteISP implementation https://github.com/cszhilu1998/RAW-to-sRGB
'''
import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm
import numpy as np
import sys
import wandb
from collections import OrderedDict
from util.util import calc_psnr as calc_psnr
import torch.backends.cudnn as cudnn
import random


if __name__ == '__main__':
	opt = TrainOptions().parse()

	seed = opt.seed
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	cudnn.benchmark = True

	wandb.init(project='your_project', entity='login', name=opt.name)
	config = wandb.config

	dataset_train = create_dataset(opt.dataset_name, 'train', opt)
	dataset_size_train = len(dataset_train)
	print('The number of training images = %d' % dataset_size_train)
	dataset_val = create_dataset(opt.dataset_name, 'val', opt)
	dataset_size_val = len(dataset_val)
	print('The number of val images = %d' % dataset_size_val)


	if opt.finetune:
		model_load = torch.load(opt.pre_path)
		model = create_model(opt)
		model.setup(opt)

		msg = model.netMetaISPNet.load_state_dict(model_load, strict=True)
		print("Loading MetaISP...")
		print(msg)

	else:
		model = create_model(opt)
		model.setup(opt)

	visualizer = Visualizer(opt)
	total_iters = ((model.start_epoch * (dataset_size_train // opt.batch_size)) \
					// opt.print_freq) * opt.print_freq

	wandb.watch(model.netMetaISPNet)
	for epoch in range(model.start_epoch + 1, opt.niter + opt.niter_decay + 1):
		# training
		epoch_start_time = time.time()
		epoch_iter = 0
		model.train()

		if opt.finetune:
			# During the fine-tuning process, we want to keep the running mean and std from the monitor. 
			# .eval guarantee that but also deactivate any dropout.
			model.netMetaISPNet.module.xcit.eval()

		iter_data_time = iter_start_time = time.time()
		for i, data in enumerate(dataset_train):
			if total_iters % opt.print_freq == 0:
				t_data = time.time() - iter_data_time
			total_iters += 1 
			epoch_iter += 1 

			model.set_input(data)
			model.optimize_parameters()

			if total_iters % opt.print_freq == 0:

				losses = model.get_current_losses()
				wandb.log({"Train Loss": losses['Total']}) 
				t_comp = (time.time() - iter_start_time)
				visualizer.print_current_losses(
					epoch, epoch_iter, losses, t_comp, t_data, total_iters)
				iter_start_time = time.time()

			iter_data_time = time.time()

		if epoch % 2 == 0:
			print('saving the model at the end of epoch %d, iters %d'
				  % (epoch, total_iters))
			model.save_networks(epoch)

		print('End of epoch %d / %d \t Time Taken: %.3f sec'
			  % (epoch, opt.niter + opt.niter_decay,
				 time.time() - epoch_start_time))
		model.update_learning_rate()

		# validating patches
		if epoch %  2 == 0:
			model.eval()
			val_iter_time = time.time()
			opt.full = False
			tqdm_val = tqdm(dataset_val)
			psnr = 0
			time_val = 0
			for i, data in enumerate(tqdm_val):
				model.set_input(data)
				time_val_start = time.time()
				with torch.no_grad():
					model.test()
				time_val += time.time() - time_val_start
				res = model.get_current_visuals()
				psnr += calc_psnr(res['dslr_warp'], res['data_out'])
			avg_psnr = psnr/len(dataset_val)
			visualizer.print_psnr(epoch, opt.niter + opt.niter_decay, time_val, avg_psnr)
			wandb.log({"PSNR": avg_psnr})
			
		sys.stdout.flush()
