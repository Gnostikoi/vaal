import pandas as pd
import numpy as np
import os
import pickle 

from sklearn.preprocessing import MultiLabelBinarizer

from .semeval5 import LoadData
from .embeddings import generate_features

class LoadSE5:
	def __init__(self, name, subtask, doc_level, model, mlb=None):
		self.root_dir = "data/SemEval_task5/sb" + str(subtask) + "/" + name + "/"
		self.doc_level = doc_level
		self.label_column = 'category'
		self.model = model
		# load dataframe
		train_df_path = self.root_dir + "train.csv"
		test_df_path = self.root_dir + "gold.csv"
		if doc_level and subtask == 2:
			train_df_path = self.root_dir + "train_dl.csv"
			test_df_path = self.root_dir + "gold_dl.csv"

		if os.path.isfile(train_df_path):
			self.train_df = pd.read_csv(train_df_path, sep='\t')
		if os.path.isfile(test_df_path):
			self.test_df = pd.read_csv(test_df_path, sep='\t')
		self.mlb = mlb

	def load_X(self, train=True, test=True):
		train_X, test_X = None, None
		if train:
			train_feature_file_path = self.root_dir + 'train_features_' + self.model
			if self.doc_level:
				train_feature_file_path = train_feature_file_path + '_dl'
		# load X
			if os.path.isfile(train_feature_file_path):
				train_X = pickle.load(open(train_feature_file_path, 'rb'))
			else:
				text = self.train_df.drop_duplicates(subset='id', keep = 'first')['sentence'].tolist()
				train_X = generate_features(self.model, text)
				pickle.dump(train_X, open(train_feature_file_path,'wb'))
		if test:
			test_feature_file_path = self.root_dir + 'test_features_' + self.model
			if self.doc_level:
				test_feature_file_path = test_feature_file_path + '_dl'

			if os.path.isfile(test_feature_file_path):
				test_X = pickle.load(open(test_feature_file_path, 'rb'))
			else:
				text = self.test_df.drop_duplicates(subset='id', keep = 'first')['sentence'].tolist()
				test_X = generate_features(self.model, text)
				pickle.dump(test_X, open(test_feature_file_path,'wb'))
		return train_X, test_X

	def load_y(self, mlb = True, train=True, test=True):
		train_y, test_y = None, None
		if train:
			train_y = self.train_df.groupby('id',sort=False)[self.label_column]\
						.agg(lambda x: list(x)).values
		if test:
			test_y = self.test_df.groupby('id',sort=False)[self.label_column]\
						.agg(lambda x: list(x)).values
		if mlb:
			if train:
				train_y = self.multi_hot_encoder(train_y)
			if test:
				test_y = self.multi_hot_encoder(test_y)	
		return train_y, test_y

	def multi_hot_encoder(self, y):
		if self.mlb is None:
			self.mlb = mlb = MultiLabelBinarizer()
			multi_y = mlb.fit_transform(y)
		else:
			multi_y = self.mlb.transform(y)
		assert multi_y.shape[0] == len(y)
		return multi_y

	def load_data(self):
		train_X, test_X = self.load_X()
		train_y, test_y = self.load_y()
			
		assert train_X.shape[0] == train_y.shape[0], f"Train X and y don't match: {train_X.shape},{train_y.shape}"
		assert test_X.shape[0] == test_y.shape[0], f"Test X and y don't match: {test_X.shape},{test_y.shape}"

		return train_X, train_y, test_X, test_y

	def load_test_data(self):
		_, test_X = self.load_X(train=False)
		_, test_y = self.load_y(train=False)
			
		assert test_X.shape[0] == test_y.shape[0], f"Test X and y don't match: {test_X.shape},{test_y.shape}"

		return test_X, test_y