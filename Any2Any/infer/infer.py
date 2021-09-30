import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
# import sys
# sys.path.append("")
import torch
from torch.backends import cudnn
import numpy as np
import yaml
from torch.utils.data import DataLoader
from Any2Any import util
from Any2Any.meldataset import Test_MelDataset, get_infer_dataset_filelist,mel_denormalize
from Any2Any.model.any2any import MagicModel
from hifivoice.inference_e2e import  hifi_infer

class Solver():
	def __init__(self, config):
		super(Solver, self).__init__()
		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.make_records()
		self.Generator = MagicModel().to(self.device)
		if self.config['pre_train_singlevc']:
			singlevc_checkpoint = torch.load(self.config['singlevc_model_path'], map_location='cpu')
			self.Generator.any2one.load_state_dict(singlevc_checkpoint['Generator'])
			self.Generator.any2one.eval()
			self.Generator.any2one.remove_weight_norm()
		self.resume_model(self.config['resume_path'])
		self.logging.info('config = %s', self.config)
		print('param Generator size = %fM ' % (util.count_parameters_in_M(self.Generator)))

	def make_records(self):
		time_record = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
		self.log_dir = os.path.join(self.config['out_dir'],time_record,"log")
		self.convt_mel_dir = os.path.join(self.config['out_dir'],time_record,"infer","mel")
		self.convt_voice_dir = os.path.join(self.config['out_dir'],time_record,"infer","voice")
		os.makedirs(self.log_dir, exist_ok=True)
		os.makedirs(self.convt_mel_dir, exist_ok=True)
		os.makedirs(self.convt_voice_dir, exist_ok=True)
		self.logging = util.Logger(self.log_dir, "log.txt")

	def get_test_data_loaders(self):
		test_filelist = get_infer_dataset_filelist(self.config["test_wav_dir"])
		testset = Test_MelDataset(test_filelist, self.config["wav2mel_model_path"],self.config["dvector_model_path"],self.config["n_fft"],self.config["num_mels"],
							 self.config["hop_size"], self.config["win_size"], self.config["sampling_rate"],self.config["fmin"],
							 self.config["fmax"], device=self.device)
		test_data_loader = DataLoader(testset, num_workers=1, shuffle=False, sampler=None,
									  batch_size=1, pin_memory=False, drop_last=True)
		return test_data_loader

	def resume_model(self,resume_path):
		print("*********  [load]   ***********")
		checkpoint_file = resume_path
		self.logging.info('loading the model from %s' % (checkpoint_file))
		checkpoint = torch.load(checkpoint_file, map_location='cpu')
		self.Generator.load_state_dict(checkpoint['Generator'])

	def infer(self):
		# infer  prepare
		test_data_loader = self.get_test_data_loaders()
		self.Generator.eval()
		self.Generator.cont_encoder.remove_weight_norm()
		self.Generator.generator.remove_weight_norm()
		mel_npy_file_list=[]
		with torch.no_grad():
			for idx, (spk_emb,input_mel, word) in enumerate(test_data_loader):
				input_mel = input_mel.cuda()
				spk_emb = spk_emb.cuda()
				fake_mel = self.Generator(spk_emb,input_mel,None)
				# fake_mel = input_mel
				fake_mel = torch.clamp(fake_mel, min=0, max=1)
				fake_mel = mel_denormalize(fake_mel)
				fake_mel = fake_mel.transpose(1,2)
				fake_mel = fake_mel.detach().cpu().numpy()

				file_name = word[0]
				mel_npy_file = os.path.join(self.convt_mel_dir, file_name+ '.npy')
				# mel_npy_list.append(mel_npy_file)
				np.save(mel_npy_file, fake_mel, allow_pickle=False)
				mel_npy_file_list.append([file_name,fake_mel])

				if len(mel_npy_file_list)==500 or idx == len(test_data_loader)-1:
					self.logging.info('【infer_%d】 len: %d', idx,len(mel_npy_file_list))
					hifi_infer(mel_npy_file_list,self.convt_voice_dir,self.config["hifi_model_path"],self.config["hifi_config_path"])
					mel_npy_file_list.clear()


	
if __name__ == '__main__':
	print("【Solver】" )
	cudnn.benchmark = True
	config_path = r"Any2Any/infer/infer_config.yaml"
	with open(config_path) as f:
		config = yaml.load(f, Loader=yaml.Loader)
	solver = Solver(config)
	solver.infer()


