import numpy as np
import os
from data.base_dataset import BaseDataset
from .imlib import imlib
from multiprocessing.dummy import Pool
from tqdm import tqdm
from util.util import augment, remove_black_level
from util.util import extract_bayer_channels
import pandas
import cv2
import random
import math

class MetaDataset(BaseDataset):
	def __init__(self, opt, split='train', dataset_name='Meta'):
		super(MetaDataset, self).__init__(opt, split, dataset_name)

		self.batch_size = opt.batch_size
		self.mode = opt.mode 
		self.imio = imlib(self.mode, lib=opt.imlib)
		self.raw_imio = imlib('RAW', fmt='HWC', lib='cv2')

		self.multi_device = opt.multi_device
		self.inference = opt.inference
		

		if opt.datatype == 'monitor':
			path_data = "datasets/monitor/"
		elif opt.datatype == 'real':
			path_data = "datasets/real/"


		if split == 'train':

			rgb_pixel = path_data+"pixel/rgb/train/"
			rgb_samsung = path_data+"samsung/rgb/train/"
			rgb_iphone = path_data+"iphone/rgb/train/"

			self.dslr_dir = [rgb_pixel,rgb_samsung,rgb_iphone]

			self.names_rgb = (pandas.read_csv(path_data+"meta/iphone/meta_train_patch.csv").to_numpy()[:,0]).tolist()


			self.raw_dir = path_data+"iphone/raw/train/"

			self.dir_raw_bilinear = path_data+"iphone/bilinear/raw/train/"

			self.meta_i = pandas.read_csv(path_data+"meta/iphone/meta_train_patch.csv").to_numpy()
			self.meta_p = pandas.read_csv(path_data+"meta/pixel/meta_train_patch.csv").to_numpy()
			self.meta_s = pandas.read_csv(path_data+"meta/samsung/meta_train_patch.csv").to_numpy()
			self.meta = [self.meta_p, self.meta_s, self.meta_i]

			self._getitem = self._getitem_train

		elif split == 'val':

			if self.opt.full:
				
				rgb_pixel = path_data+"pixel/full/rgb/val/"
				rgb_samsung = path_data+"samsung/full/rgb/val/"
				rgb_iphone = path_data+"iphone/full/rgb/val/"

				self.dslr_dir = [rgb_pixel,rgb_samsung,rgb_iphone]

				self.names_rgb = (pandas.read_csv(path_data+"meta/iphone/meta_val.csv").to_numpy()[:,0]).tolist()


				self.raw_dir = path_data+"iphone/full/raw/val/"

				self.dir_raw_bilinear = path_data+"iphone/bilinear/raw/val/"


				self.meta_i = pandas.read_csv(path_data+"meta/iphone/meta_val.csv").to_numpy()
				self.meta_p = pandas.read_csv(path_data+"meta/pixel/meta_val.csv").to_numpy()
				self.meta_s = pandas.read_csv(path_data+"meta/samsung/meta_val.csv").to_numpy()
				self.meta = [self.meta_p, self.meta_s, self.meta_i]

				self._getitem = self._getitem_test
			else:

				rgb_pixel = path_data+"pixel/rgb/val/"
				rgb_samsung = path_data+"samsung/rgb/val/"
				rgb_iphone = path_data+"iphone/rgb/val/"

				self.dslr_dir = [rgb_pixel,rgb_samsung,rgb_iphone]

				self.names_rgb = (pandas.read_csv(path_data+"meta/iphone/meta_val_patch.csv").to_numpy()[:,0]).tolist()


				self.raw_dir = path_data+"iphone/raw/val/"

				self.dir_raw_bilinear = path_data+"iphone/bilinear/raw/val/"


				self.meta_i = pandas.read_csv(path_data+"meta/iphone/meta_val_patch.csv").to_numpy()
				self.meta_p = pandas.read_csv(path_data+"meta/pixel/meta_val_patch.csv").to_numpy()
				self.meta_s = pandas.read_csv(path_data+"meta/samsung/meta_val_patch.csv").to_numpy()
				self.meta = [self.meta_p, self.meta_s, self.meta_i]

				self._getitem = self._getitem_test

		elif split == 'test':

			if self.opt.full:
				
				rgb_pixel = path_data+"pixel/full/rgb/test/"
				rgb_samsung = path_data+"samsung/full/rgb/test/"
				rgb_iphone = path_data+"iphone/full/rgb/test/"

				self.dslr_dir = [rgb_pixel,rgb_samsung,rgb_iphone]

				self.names_rgb = (pandas.read_csv(path_data+"meta/iphone/meta_test.csv").to_numpy()[:,0]).tolist()

				self.raw_dir = path_data+"iphone/full/raw/test/"
				self.dir_raw_bilinear = path_data+"iphone/bilinear/raw/test/"


				self.meta_i = pandas.read_csv(path_data+"meta/iphone/meta_test.csv").to_numpy()
				self.meta_p = pandas.read_csv(path_data+"meta/pixel/meta_test.csv").to_numpy()
				self.meta_s = pandas.read_csv(path_data+"meta/samsung/meta_test.csv").to_numpy()
				self.meta = [self.meta_p, self.meta_s, self.meta_i]
			else:

				rgb_pixel = path_data+"pixel/rgb/test/"
				rgb_samsung = path_data+"samsung/rgb/test/"
				rgb_iphone = path_data+"iphone/rgb/test/"

				self.dslr_dir = [rgb_pixel,rgb_samsung,rgb_iphone]

				self.names_rgb = (pandas.read_csv(path_data+"meta/iphone/meta_test_patch.csv").to_numpy()[:,0]).tolist()


				self.raw_dir = path_data+"iphone/raw/test/"


				self.dir_raw_bilinear = path_data+"iphone/bilinear/raw/test/"
				self.meta_i = pandas.read_csv(path_data+"meta/iphone/meta_test_patch.csv").to_numpy()
				self.meta_p = pandas.read_csv(path_data+"meta/pixel/meta_test_patch.csv").to_numpy()
				self.meta_s = pandas.read_csv(path_data+"meta/samsung/meta_test_patch.csv").to_numpy()
				self.meta = [self.meta_p, self.meta_s, self.meta_i]
				self._getitem = self._getitem_test

	
			self._getitem = self._getitem_test
		self.names = self.names_rgb
		self.len_data = len(self.names)
		self.raw_images = [0] * self.len_data
		self.patch_size = 448
		self.full_size = 2688

	def __getitem__(self, index):
		return self._getitem(index)

	def __len__(self):
		return self.len_data

	def _getitem_train(self, idx):
		raw_combined = self._process_raw(self.raw_images[idx],idx)

		if self.opt.individual:	
			idx_dataset = self.opt.infedev
		else:
			idx_dataset = random.randint(0,2)
		dslr_image_ref = self.imio.read(os.path.join(self.dslr_dir[2], self.names_rgb[idx][:-4] + ".png"))
		dslr_image_ref = np.float32(dslr_image_ref) / 255.0

		if idx_dataset != 2:
			dslr_image = self.imio.read(os.path.join(self.dslr_dir[idx_dataset], self.names_rgb[idx][:-4] + ".png"))
			dslr_image = np.float32(dslr_image) / 255.0
		else:
			dslr_image = dslr_image_ref


		if self.opt.illuminant:
			r_meta = self.meta[idx_dataset][idx,1]
			g_meta = self.meta[idx_dataset][idx,2]
			b_meta = self.meta[idx_dataset][idx,3]
			wb = np.expand_dims(np.expand_dims(np.array([b_meta,g_meta,r_meta,g_meta]), 1),1)
			iso = self.meta[2][idx][5]/100
			exp = self.meta[2][idx][4]*100
		else:
			r_meta = self.meta[2][idx,1]
			g_meta = self.meta[2][idx,2]
			b_meta = self.meta[2][idx,3]
			wb = np.expand_dims(np.expand_dims(np.array([b_meta,g_meta,r_meta,g_meta]), 1),1)
			iso = self.meta[2][idx][5]/100
			exp = self.meta[2][idx][4]*100

		raw_down_name = self.names[idx].rsplit('_', 1)[0]+'.png'
		raw_down = cv2.imread(self.dir_raw_bilinear+raw_down_name, cv2.IMREAD_UNCHANGED)
		raw_down = raw_down.astype(np.float32)#/(16 * 255)
		raw_down = remove_black_level(raw_down)
		raw_down = raw_down.transpose((2, 0, 1))
		
		patch_number = int(self.names[idx].rsplit('_', 1)[1][:-4])
		p_x = patch_number%6
		p_y = patch_number//6
		s_x = (self.patch_size//2*p_x)+1
		f_x = (self.patch_size//2) + self.patch_size//2*p_x
		s_y = (self.patch_size//2*p_y)+1
		f_y = (self.patch_size//2) + self.patch_size//2*p_y
		xs = np.linspace(s_x, f_x, num=(self.patch_size//2))
		ys = np.linspace(s_y, f_y, num=(self.patch_size//2))
		x, y = np.meshgrid(xs, ys, indexing='xy')
		y = np.expand_dims(y, axis=0)
		x = np.expand_dims(x, axis=0)
		coords = np.ascontiguousarray(np.concatenate([x, y]))

		scale = 2 * math.pi
		coords[1,:,:] = coords[1,:,:] / (self.full_size/2) * scale 
		coords[0,:,:] = coords[0,:,:] / (self.full_size/2) * scale

		raw_combined,  dslr_image, raw_down, dslr_image_ref, coords = augment(raw_combined,  dslr_image, raw_down, dslr_image_ref, coords)
	
		if self.multi_device:
			out = {'raw': raw_combined,
				'dslr': dslr_image,
				'fname': self.names[idx],
				'raw_demosaic_full': raw_down,
				'wb':wb,
				'dslr_image_ref':dslr_image_ref,
				'device':idx_dataset,
				'coords':coords,
				'exp':exp,
				'iso':iso}
		else:
			out = {'raw': raw_combined,
				'dslr': dslr_image,
				'fname': self.names[idx],
				'raw_demosaic_full': raw_down,
				'dslr_image_ref':dslr_image_ref,
				'wb':wb,
				'coords':coords,
				'exp':exp,
				'iso':iso}

		return out

	def _getitem_test(self, idx):
		raw_combined = self._process_raw(self.raw_images[idx], idx)
		if not self.opt.individual:	
			if self.opt.full:
				idx_dataset = self.opt.infedev
			else:
				if self.inference:
					idx_dataset = self.opt.infedev
				else:
					idx_dataset = random.randint(0,2)
				
			dslr_image = self.imio.read(os.path.join(self.dslr_dir[idx_dataset], self.names_rgb[idx][:-4] + ".png"))
			dslr_image = np.float32(dslr_image) / 255.0	

			dslr_image_ref = self.imio.read(os.path.join(self.dslr_dir[2], self.names_rgb[idx][:-4] + ".png"))
			dslr_image_ref = np.float32(dslr_image_ref) / 255.0

			if self.opt.illuminant:
				r_meta = self.meta[idx_dataset][idx,1]
				g_meta = self.meta[idx_dataset][idx,2]
				b_meta = self.meta[idx_dataset][idx,3]
				wb = np.expand_dims(np.expand_dims(np.array([b_meta,g_meta,r_meta,g_meta]), 1),1)
				iso = self.meta[2][idx][5]/100
				exp = self.meta[2][idx][4]*100
			else:
				r_meta = self.meta[2][idx,1]
				g_meta = self.meta[2][idx,2]
				b_meta = self.meta[2][idx,3]
				iso = self.meta[2][idx][5]/100
				exp = self.meta[2][idx][4]*100
				
				wb = np.expand_dims(np.expand_dims(np.array([b_meta,g_meta,r_meta,g_meta]), 1),1)
			
		else:

			idx_dataset = self.opt.infedev

				
			dslr_image = self.imio.read(os.path.join(self.dslr_dir[idx_dataset], self.names_rgb[idx][:-4] + ".png"))
			dslr_image = np.float32(dslr_image) / 255.0	

			dslr_image_ref = self.imio.read(os.path.join(self.dslr_dir[2], self.names_rgb[idx][:-4] + ".png"))
			dslr_image_ref = np.float32(dslr_image_ref) / 255.0

			if self.opt.illuminant:
				r_meta = self.meta[idx_dataset][idx,1]
				g_meta = self.meta[idx_dataset][idx,2]
				b_meta = self.meta[idx_dataset][idx,3]
				wb = np.expand_dims(np.expand_dims(np.array([b_meta,g_meta,r_meta,g_meta]), 1),1)
				iso = self.meta[2][idx][5]/100
				exp = self.meta[2][idx][4]*100
			else:
				r_meta = self.meta[2][idx,1]
				g_meta = self.meta[2][idx,2]
				b_meta = self.meta[2][idx,3]
				iso = self.meta[2][idx][5]/100
				exp = self.meta[2][idx][4]*100
				
				wb = np.expand_dims(np.expand_dims(np.array([b_meta,g_meta,r_meta,g_meta]), 1),1)

		if self.opt.full:
			raw_down_name = self.names[idx][:-4]+'.png'
			xs = np.linspace(1, self.full_size//2, num=self.full_size//2)
			ys = np.linspace(1, self.full_size//2, num=self.full_size//2)
			x, y = np.meshgrid(xs, ys, indexing='xy')
			y = np.expand_dims(y, axis=0)
			x = np.expand_dims(x, axis=0)
			coords = np.ascontiguousarray(np.concatenate([x, y]))
			scale = 2 * math.pi
			coords[1,:,:] = coords[1,:,:] / (self.full_size/2) * scale 
			coords[0,:,:] = coords[0,:,:] / (self.full_size/2) * scale
		else:
			patch_number = int(self.names[idx].rsplit('_', 1)[1][:-4])
			p_x = patch_number%6
			p_y = patch_number//6
			s_x = (self.patch_size//2*p_x)+1
			f_x = self.patch_size//2 + self.patch_size//2*p_x
			s_y = (self.patch_size//2*p_y)+1
			f_y = self.patch_size//2 + self.patch_size//2*p_y
			xs = np.linspace(s_x, f_x, num=self.patch_size//2)
			ys = np.linspace(s_y, f_y, num=self.patch_size//2)
			x, y = np.meshgrid(xs, ys, indexing='xy')
			y = np.expand_dims(y, axis=0)
			x = np.expand_dims(x, axis=0)
			coords = np.ascontiguousarray(np.concatenate([x, y]))

			eps = 1e-6
			scale = 2 * math.pi
			coords[1,:,:] = coords[1,:,:] / (self.full_size/2) * scale 
			coords[0,:,:] = coords[0,:,:] / (self.full_size/2) * scale
			raw_down_name = self.names[idx].rsplit('_', 1)[0]+'.png'#.rsplit('_', 1)[0]+'.png'
			
		raw_down = cv2.imread(self.dir_raw_bilinear+raw_down_name, cv2.IMREAD_UNCHANGED)
		raw_down = raw_down.astype(np.float32)
		raw_down = remove_black_level(raw_down)
		raw_down = raw_down.transpose((2, 0, 1))

		if self.multi_device:
			out = {'raw': raw_combined,
				'dslr': dslr_image,
				'fname': self.names[idx],
				'raw_demosaic_full': raw_down,
				'wb':wb,
				'dslr_image_ref': dslr_image_ref,
				'device':idx_dataset,
				'coords':coords,
				'iso':iso,
				'exp':exp}
		else:
			out = {'raw': raw_combined,
				'dslr': dslr_image,
				'fname': self.names[idx],
				'dslr_image_ref': dslr_image_ref,
				'raw_demosaic_full': raw_down,
				'wb':wb,
				'coords':coords,
				'iso':iso,
				'exp':exp}

		return out

	def _process_raw(self, raw, idx):
		raw = self.raw_imio.read(os.path.join(self.raw_dir, self.names[idx][:-4] + ".png"))
		raw = remove_black_level(raw)
		raw_combined = extract_bayer_channels(raw)

		return raw_combined

def iter_obj(num, objs):
	for i in range(num):
		yield (i, objs)

def imreader(arg):
	i, obj = arg
	for _ in range(3):
		try:
			obj.raw_images[i] = obj.raw_imio.read(os.path.join(obj.raw_dir, obj.names[i][:-4] + '.png'))
			failed = False
			break
		except:
			failed = True
	if failed: print('%s fails!' % obj.names[i])

def read_images(obj):
	print('Starting to load images via multiple imreaders')
	pool = Pool() # use all threads by default
	for _ in tqdm(pool.imap(imreader, iter_obj(obj.len_data, obj)), total=obj.len_data):
		pass
	pool.close()
	pool.join()

if __name__ == '__main__':
	pass
