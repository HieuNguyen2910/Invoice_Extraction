import argparse
from utils import load_config, crop_background, measure, Progress
import cv2
import os
from time import time
from PIL import Image
from multiprocessing import Pool
from rembg import remove
from rotation import model, Craft, align_box, rotate_90, rotate_180
from text_extraction import Config, Predictor
import numpy as np
import torch


class Pipeline:
	"""Run file pipeline"""
	def __init__(self, config):
		self.config = config
		self.data = None
		self.text_detector = None
		self.text_extractor = None

	def load_image(self, image_path):
		"""Load the image"""
		image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
		height = self.config['image_size']
		width = int(image.shape[1]*(height/image.shape[0]))
		return cv2.resize(image, (width, height))  # resize image

	@measure
	def prepare_data(self):
		"""Prepare the dataset"""
		if not os.path.exists(self.config["input"]):
			print(f'[Error] No such file or directory: {self.config["input"]}')
			os._exit(0)
		if not os.path.exists(self.config["output"]):
			os.mkdir(self.config["output"])

		if os.path.isfile(self.config["input"]):  # load if input is file
			filename = self.config["input"].split('/')[-1]
			self.data = [{
				'name': filename.split('.')[0],
				'image': self.load_image(self.config["input"])
				}]
			self.config["input"] = self.config["input"].replace(filename, '')
		else:  # load if input is folder
			files = os.listdir(self.config["input"])
			self.data = [{
				'name': filename.split('.')[0],
				'image': self.load_image(f'{self.config["input"]}/{filename}')
				} for filename in files]
		return self.data

	# @staticmethod
	# def remove_background(img_data):
	# 	"""Remove background"""
	# 	bg_removed = remove(img_data['image'])
	# 	img_data['image'] = crop_background(bg_removed)
	# 	return img_data

	@measure
	def prepare_model(self):
		"""Prepare the model for gpu or cpu"""
		ocr_config = Config.load_config_from_name(self.config['vietocr_model'])
		if (self.config['gpu'] != 0) and torch.cuda.is_available():
			self.text_detector = Craft('cuda')
			ocr_config['device'] = 'cuda' if self.config['gpu'] == -1 else f"cuda:{self.config['gpu']-1}"
		else:
			self.text_detector = Craft('cpu')
			ocr_config['device'] = 'cpu'
		self.text_extractor = Predictor(ocr_config)

	def rotate(self, img_data):
		"""Skip rotation, only detect text boxes"""
		img = model.loadImage(img_data['image'])
		bboxes = self.text_detector(img)
		img_data['image'] = img
		img_data['bboxes'] = bboxes
		return img_data

	def extract_info(self, img_data):
		"""Extract information"""
		img_data['information'] = []
		incline = {'prev_height': 0, 'prev_line': -1, }
		for i, box in enumerate(img_data['bboxes']):
			x1 = int(box[0][0] if (box[0][0] < box[3][0]) else box[3][0])
			y1 = int(box[0][1] if (box[0][1] < box[1][1]) else box[1][1])
			x2 = int(box[2][0] if (box[2][0] > box[1][0]) else box[1][0])
			y2 = int(box[2][1] if (box[2][1] > box[3][1]) else box[3][1])

			arr_img = img_data['image'].copy()[y1:y2, x1:x2]  # crop image
			if arr_img.size == 0 or arr_img.shape[0] == 0 or arr_img.shape[1] == 0:
				continue

			try:
				img_box = Image.fromarray(arr_img)
				detected = self.text_extractor.predict(img_box)  # OCR
			except:
				continue

			# lưu box + text
			if self.config['incline']:
				current_height = sum(y[1] for y in box )/4
				per_diff = abs(1-incline['prev_height']/current_height)
				if per_diff < 0.02:  
					img_data['information'][incline['prev_line']].append((box, detected))
				else:
					img_data['information'].append([(box, detected)])
					incline['prev_line'] += 1
				incline['prev_height'] = current_height
			else:
				img_data['information'].append([(box, detected)])
		return img_data

	def save_image(self, img_data):
		"""Save original image into train/img folder"""
		if not self.config['save_image']:
			return

		# Tạo folder train/img nếu chưa tồn tại
		img_folder = os.path.join(self.config['output'], "train", "img")
		os.makedirs(img_folder, exist_ok=True)

		# Lưu ảnh gốc (grayscale -> RGB)
		if len(img_data['image'].shape) == 2:
			image = cv2.cvtColor(img_data['image'], cv2.COLOR_GRAY2RGB)
		else:
			image = img_data['image']
		image_path = os.path.join(img_folder, f"{img_data['name']}.jpg")
		Image.fromarray(image).save(image_path)


	def save_text(self, img_data):
		"""Save coordinates + transcript and create empty entities JSON"""
		if not self.config['save_text']:
			return

		# Tạo folder train/box và train/entities nếu chưa tồn tại
		box_folder = os.path.join(self.config['output'], "train", "box")
		entities_folder = os.path.join(self.config['output'], "train", "entities")
		os.makedirs(box_folder, exist_ok=True)
		os.makedirs(entities_folder, exist_ok=True)

		# Lưu file tọa độ + text vào folder box
		box_path = os.path.join(box_folder, f"{img_data['name']}.txt")
		with open(box_path, "w+", encoding="utf-8") as f:
			for line in img_data['information']:
				for box, transcript in line:
					coords = ",".join([f"{int(x)},{int(y)}" for x, y in box])
					f.write(f"{coords},{transcript}\n")

		# Tạo file entities JSON rỗng để tự điền nhãn
		entities_path = os.path.join(entities_folder, f"{img_data['name']}.txt")
		empty_entities = {
			"company": "",
			"date": "",
			"address": "",
			"total": ""
		}
		import json
		with open(entities_path, "w+", encoding="utf-8") as f:
			json.dump(empty_entities, f, ensure_ascii=False, indent=4)
@measure
def main(args):
	config = load_config('run', args)  # load config

	pl = Pipeline(config)

	data = pl.prepare_data()

	pl.prepare_model()

	start = time()
	# if config['multiprocessing'] in [0, 1]:  # multiprocessing disable
	# 	print(f'Multiprocessing will not be used!')
	# 	bg_removed = [pl.remove_background(img_data) for img_data in data]
	# else:  # multiprocessing enable
	# 	max_cpu = int(torch.multiprocessing.cpu_count()*0.8)  # 80% for safety | max out your thread may crash your system
	# 	num_cpu = max_cpu if config['multiprocessing'] == -1 else config['multiprocessing']
	# 	print(f'Maximum {num_cpu} cpu will be used')
	# 	with Pool(processes=num_cpu) as pool:
	# 		bg_removed = pool.map(pl.remove_background, data)
	# print(f'Done remove background in {round(time()-start, 2)}s')

	print('Start extract information...')
	for img_data in Progress(data):  # extract information
		img_data = pl.rotate(img_data)
		img_data = pl.extract_info(img_data)
		pl.save_text(img_data)
		pl.save_image(img_data)
	print(f"Result has been saved to '{config['output']}'")


if __name__ == '__main__':
	args = argparse.ArgumentParser(description='Extract Receipt Information')  # setup execution argument
	args.add_argument('-i', '--input', type=str, help='Image path or Folder path (Default: data/test/)')
	args.add_argument('-o', '--output', type=str, help='Output folder path (Default: result/)')
	args.add_argument('-g', '--gpu', type=int, help='Use which gpu | 0 for cpu | -1 for all (Default: -1)')
	args.add_argument('-mp', '--multiprocessing', type=int, help='Maximum of cpu can use | -1 for 80 percent (Default: -1)')
	args = args.parse_args()

	main(args)
