import xml.etree.ElementTree as ET
import numpy as np
import os 
import pandas as pd


class LoadData:
	def __init__(self, root_dir):
		self.root_dir = root_dir

	def load(self, sb, name, sen_level = False, persist=True):
		path = self.root_dir + "sb" + str(sb)+ "/" + name + '/'		
		if sb == 1:
			train_df = self.parse_sb1(path+"train.xml")
			test_df = self.parse_sb1(path+"gold.xml")
		elif sb == 2:
			train_df = self.parse_sb2(path+"train.xml", sen_level)
			test_df = self.parse_sb2(path+"gold.xml", sen_level)
		if persist:
			if not sen_level and sb == 2:
				if train_df is not None:
					train_df.to_csv(path+"train_dl.csv", sep='\t')
				if test_df is not None:
					test_df.to_csv(path+"gold_dl.csv", sep='\t')
			else:
				if train_df is not None:
					train_df.to_csv(path+"train.csv", sep='\t')
				if test_df is not None:
					test_df.to_csv(path+"gold.csv", sep='\t')
		return train_df, test_df


	def parse_sb1(self, filepath):
		if not os.path.isfile(filepath):
			return None
		tree = ET.parse(filepath)
		root = tree.getroot()	
		df = pd.DataFrame()

		for sentence in root.iter("sentence"):
			sen_id = sentence.get("id")
			text = sentence.find("text")
			opinions = sentence.find("Opinions")
			
			if text is not None and opinions is not None:
				t = {"sentence": text.text, "id": sen_id}
				for opinion in opinions.findall("Opinion"):
					data = {**opinion.attrib, **t}
					data = pd.DataFrame(data, index=[0])
					df = df.append(data, ignore_index=True)
		return df

	def parse_sb2(self, filepath, level):
		if not os.path.isfile(filepath):
			return None
		tree = ET.parse(filepath)
		root = tree.getroot()	
		df = pd.DataFrame()
		for review in root.iter("Review"):
			rev_id = review.get("rid")
			doc = [t.text for t in review.iter("text")]
			sen_id = [sen.get("id") for sen in review.iter("sentence")]
			if not level:
				doc = " ".join(doc)
			else:
				doc = list(zip(doc, sen_id))
			opinions = review.find("Opinions")
			if opinions is not None:
				for opinion in opinions.findall("Opinion"):
					if level:
						for (text, sen_id) in doc:
							t = {"sentence": text, "id": sen_id, "rid": rev_id}
							data = {**opinion.attrib, **t}
							data = pd.DataFrame(data, index=[0])
							df = df.append(data, ignore_index = True)
					else:
						t = {"sentence": doc, "id": rev_id}
						data = {**opinion.attrib, **t}
						data = pd.DataFrame(data, index=[0])
						df = df.append(data, ignore_index = True)
		return df
