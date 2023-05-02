import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import torch
import pandas as pd
import torch.utils.data as data_utils
import os
import glob
from skimage import io
from tqdm import tqdm
import h5py


# def statistics_slide(slide_path_list):
#     num_pos_patch_allPosSlide = 0
#     num_patch_allPosSlide = 0
#     num_neg_patch_allNegSlide = 0
#     num_all_slide = len(slide_path_list)

#     for i in slide_path_list:
#         if 'pos' in i.split('/')[-1]:  # pos slide
#             num_pos_patch = len(glob.glob(i + "/*_pos.jpg"))
#             num_patch = len(glob.glob(i + "/*.jpg"))
#             num_pos_patch_allPosSlide = num_pos_patch_allPosSlide + num_pos_patch
#             num_patch_allPosSlide = num_patch_allPosSlide + num_patch
#         else:  # neg slide
#             num_neg_patch = len(glob.glob(i + "/*.jpg"))
#             num_neg_patch_allNegSlide = num_neg_patch_allNegSlide + num_neg_patch

#     print("[DATA INFO] {} slides totally".format(num_all_slide))
#     print("[DATA INFO] pos_patch_ratio in pos slide: {:.4f}({}/{})".format(
#         num_pos_patch_allPosSlide / num_patch_allPosSlide, num_pos_patch_allPosSlide, num_patch_allPosSlide))
#     print("[DATA INFO] num of patches: {} ({} from pos slide, {} from neg slide)".format(
#         num_patch_allPosSlide+num_neg_patch_allNegSlide, num_patch_allPosSlide, num_neg_patch_allNegSlide))
#     return num_patch_allPosSlide+num_neg_patch_allNegSlide


class BRCA_feat(torch.utils.data.Dataset):
	# @profile
	def __init__(self, csv_dir='', feat_dir='', pkl_path=None,
				 partition='train', transform=None, downsample=1.0, drop_threshold=0.0, preload=True, return_bag=False):
		# self.train = train
		self.csv_dir = csv_dir
		self.feat_dir = feat_dir
		self.transform = transform
		self.downsample = downsample
		self.drop_threshold = drop_threshold  # drop the pos slide of which positive patch ratio less than the threshold
		self.preload = preload
		self.return_bag = return_bag
		#TODO change path
		# if train:
		#     self.root_dir = os.path.join(self.root_dir, "training")
		# else:
		#     self.root_dir = os.path.join(self.root_dir, "testing")

		slide_data = pd.read_csv(csv_dir)
		self.slides_data = slide_data[slide_data['dataset'] == partition]
		self.slides_data.reset_index(inplace=True)
		self.all_slides = self.slides_data['slide_path'].tolist()

		#TODO: comment out the filtering process since we do not have patch-level annotations
		# 1.filter the pos slides which have 0 pos patch
		
		# all_pos_slides = glob.glob(self.root_dir + "/*_pos")

		# for i in all_pos_slides:
		#     num_pos_patch = len(glob.glob(i + "/*_pos.jpg"))
		#     num_patch = len(glob.glob(i + "/*.jpg"))
		#     if num_pos_patch/num_patch <= self.drop_threshold:
		#         all_slides.remove(i)
		#         print("[DATA] {} of positive patch ratio {:.4f}({}/{}) is removed".format(
		#             i, num_pos_patch/num_patch, num_pos_patch, num_patch))
		# statistics_slide(all_slides)
		
		# 1.1 down sample the slides
		# print("================ Down sample ================")
		# np.random.shuffle(all_slides)
		# all_slides = all_slides[:int(len(all_slides)*self.downsample)]
		# self.num_slides = len(all_slides)
		#self.num_patches = statistics_slide(all_slides)

		# 2. load all pre-trained patch features (by SimCLR in DSMIL)
		all_slides_name = self.slides_data['slide'].tolist()
		# if train:
		#     all_slides_feat_file = glob.glob("")
		# else:
		#     all_slides_feat_file = glob.glob("")

		# Scanning & loading all patch features and build slide-patch mapping
		if pkl_path is None:
			self.num_patches = 0
			self.cached_bag_index = -1
			self.cached_bag = None
			self.patch_feat_all = []
			self.patch_corresponding_slide_label = []
			self.patch_corresponding_slide_index = []
			self.patch_corresponding_slide_name = []
			self.patch_corresponding_slide_intra_index = []
			for i, slide in tqdm(self.slides_data.iterrows(), total=self.slides_data.shape[0], desc='loading patches'):
				full_path = os.path.join(feat_dir, 'patches', slide['slide'] + '.h5')
				with h5py.File(full_path,'r') as hdf5_file:
					coords = hdf5_file['coords'][:]
				num_patch_in_bag = len(coords)
				self.num_patches += num_patch_in_bag
	#            self.patch_feat_all.extend(features)
				self.patch_corresponding_slide_label.extend( [slide['stage']] * num_patch_in_bag )
				self.patch_corresponding_slide_index.extend( [i] * num_patch_in_bag )
				self.patch_corresponding_slide_name.extend( [slide['slide']] * num_patch_in_bag )
				self.patch_corresponding_slide_intra_index.extend( np.arange(num_patch_in_bag) )
				
			#self.patch_feat_all = np.array(self.slide_feat_all).astype(np.float32)
			self.patch_corresponding_slide_index = np.array(self.patch_corresponding_slide_index).astype(np.int_)
			with open('patch_slide_index_{}.pkl'.format(partition), 'wb') as f:
				pkl.dump(self.patch_corresponding_slide_index, f)
			self.patch_corresponding_slide_label = np.array(self.patch_corresponding_slide_label).astype(np.int_)
			self.patch_corresponding_slide_name = np.array(self.patch_corresponding_slide_name)
			self.patch_corresponding_slide_intra_index = np.concatenate(self.patch_corresponding_slide_intra_index).astype(np.int_)
		else:
			with open(pkl_path, 'rb') as f:
				self.patch_corresponding_slide_index = pkl.load(f)
			self.num_patches = len(self.patch_corresponding_slide_index)
			slide_label_array = self.slides_data['stage'].to_numpy()
			assert self.patch_corresponding_slide_index.max() < slide_label_array.shape[0]
			self.patch_corresponding_slide_label = np.take(slide_label_array, self.patch_corresponding_slide_index)
			self.patch_corresponding_slide_name = np.take(all_slides_name, self.patch_corresponding_slide_index)
			self.patch_corresponding_slide_intra_index = np.arange(self.num_patches)
			_, counts = np.unique(self.patch_corresponding_slide_index, return_counts=True)
			self.patch_corresponding_slide_intra_index = np.concatenate([np.arange(count) for count in counts])
			assert self.patch_corresponding_slide_intra_index.shape[0] == self.num_patches

		self.patch_label_all = np.zeros([self.num_patches], dtype=np.int_)  # Patch label is not available and set to 0 !
		

		print("[DATA INFO] num_slide is {}; num_patches is {}; patch label shape is {}\n".format(len(self.all_slides), self.num_patches, self.patch_corresponding_slide_label.shape))

		# self.slide_feat_all = np.zeros([self.num_patches, 1024], dtype=np.float32)
		# self.slide_patch_label_all = np.zeros([self.num_patches], dtype=np.int_)
		# self.patch_corresponding_slide_label = np.zeros([self.num_patches], dtype=np.int_)
		# self.patch_corresponding_slide_index = np.zeros([self.num_patches], dtype=np.int_)
		# self.patch_corresponding_slide_name = np.zeros([self.num_patches], dtype='<U13')
		# cnt_slide = 0
		# pointer = 0
		# for i in all_slides_feat_file:
		#     slide_name_i = i.split('/')[-1].split('.')[0]
		#     if slide_name_i not in all_slides_name:
		#         continue
		#     slide_i_label_feat = np.load(i)
		#     slide_i_patch_label = slide_i_label_feat[:, 0]
		#     slide_i_feat = slide_i_label_feat[:, 1:]
		#     num_patches_i = slide_i_label_feat.shape[0]

		#     self.slide_feat_all[pointer:pointer+num_patches_i, :] = slide_i_feat
		#     self.slide_patch_label_all[pointer:pointer+num_patches_i] = slide_i_patch_label
		#     self.patch_corresponding_slide_label[pointer:pointer+num_patches_i] = int('pos' in slide_name_i) * np.ones([num_patches_i], dtype=np.int_)
		#     self.patch_corresponding_slide_index[pointer:pointer+num_patches_i] = cnt_slide * np.ones([num_patches_i], dtype=np.int_)
		#     self.patch_corresponding_slide_name[pointer:pointer+num_patches_i] = np.array(slide_name_i).repeat(num_patches_i)
		#     pointer = pointer + num_patches_i
		#     cnt_slide = cnt_slide + 1

		# self.all_patches = self.slide_feat_all
		# self.patch_label = self.slide_patch_label_all
		print("")

		# # 3.do some statistics
		# print("[DATA INFO] num_slide is {}; num_patches is {}\npos_patch_ratio is {:.4f}".format(
		#     self.num_slides, self.num_patches, 1.0*self.patch_label.sum()/self.patch_label.shape[0]))
		# print("")

	def __getitem__(self, index):
		if self.return_bag:
			idx_patch_from_slide_i = np.where(self.patch_corresponding_slide_index == index)[0]
#            bag = self.patch_feat_all[idx_patch_from_slide_i, :]
			slide_id = self.slides_data['slide'][index]
			full_path = os.path.join(self.feat_dir, 'resnet50-features', 'pt_files', slide_id+'.pt')
			bag = torch.load(full_path)
			
			patch_labels = self.patch_label_all[idx_patch_from_slide_i]  # Patch label is not available and set to 0 !
			slide_label = self.patch_corresponding_slide_label[idx_patch_from_slide_i][0]
			slide_index = self.patch_corresponding_slide_index[idx_patch_from_slide_i][0]
			slide_name = self.patch_corresponding_slide_name[idx_patch_from_slide_i][0]

			# check data
			if self.patch_corresponding_slide_label[idx_patch_from_slide_i].max() != self.patch_corresponding_slide_label[idx_patch_from_slide_i].min():
				raise
			if self.patch_corresponding_slide_index[idx_patch_from_slide_i].max() != self.patch_corresponding_slide_index[idx_patch_from_slide_i].min():
				raise
			return bag, [patch_labels, slide_label, slide_index, slide_name], index
		else:
			slide_index = self.patch_corresponding_slide_index[index]
			patch_bag_index = self.patch_corresponding_slide_intra_index[index]
			if slide_index != self.cached_bag_index:
				full_path = os.path.join(self.feat_dir, 'resnet50-features', 'pt_files', self.patch_corresponding_slide_name[index]+'.pt')
				self.cached_bag = torch.load(full_path)
			patch_image = self.cached_bag[patch_bag_index]
			patch_corresponding_slide_label = self.patch_corresponding_slide_label[index]
			patch_corresponding_slide_index = self.patch_corresponding_slide_index[index]
			patch_corresponding_slide_name = self.patch_corresponding_slide_name[index]

			patch_label = self.patch_label_all[index]  # Patch label is not available and set to 0 !
			return patch_image, [patch_label, patch_corresponding_slide_label, patch_corresponding_slide_index,
								 patch_corresponding_slide_name], index


	def __len__(self):
		if self.return_bag:
			return self.patch_corresponding_slide_index.max() + 1
		else:
			return self.num_patches


if __name__ == '__main__':
	# train_ds = BRCA_feat(partition='train', csv_dir='annotations.csv', feat_dir='/home/ngsci/datasets/brca-psj-path/contest-phase-2/clam-preprocessing-train', pkl_path='patch_slide_index_train.pkl', transform=None, downsample=1, drop_threshold=0, preload=True, return_bag=True)
	# train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
	# for data, label, selected in train_loader:
	# 	print(data.shape, label[0].shape, label[1].shape, label[2].shape)

	train_ds = BRCA_feat(partition='holdout', csv_dir='annotations.csv', feat_dir='/home/ngsci/datasets/brca-psj-path/contest-phase-2/clam-preprocessing-holdout', transform=None, downsample=1, drop_threshold=0, preload=True, return_bag=False)
	train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
	for data, label, selected in train_loader:
		print(data.shape, label[0].shape, label[1].shape, label[2].shape)
