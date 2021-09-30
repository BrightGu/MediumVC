import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import time
import sys
sys.path.append("/home/gyw/workspace/program/VC/MediumVC_G")
import torch
from torch.backends import cudnn
import numpy as np
import yaml

from torch.utils.data import DataLoader


from Any2Any.meldataset import Test_MelDataset,PickleDataset,get_test_dataset_filelist,get_data_loader,mel_denormalize
from Any2Any import util
from Any2Any.model.any2any import MagicModel

from hifivoice.inference_e2e import hifi_infer

class Solver():
	def __init__(self, config):
		super(Solver, self).__init__()
		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		# record
		self.make_records()
		# train
		self.total_epochs = self.config['epochs']
		self.save_period = self.config['save_period']
		self.eval_period = self.config['eval_period']
		self.step_record_time = self.config['step_record_time']
		self.learning_rate = self.config["learning_rate"]

		self.train_data_loader = self.get_data_loaders(self.config['figure_mel_label_dir'],self.config['batch_size'])
		self.len_train_data = len(self.train_data_loader)

		self.Generator = MagicModel().to(self.device)
		if self.config['pre_train_singlevc']:
			singlevc_checkpoint = torch.load(self.config['singlevc_model_path'], map_location='cpu')
			self.Generator.any2one.load_state_dict(singlevc_checkpoint['Generator'])
			self.Generator.any2one.eval()
			self.Generator.any2one.remove_weight_norm()

		for p in self.Generator.any2one.parameters():
			p.requires_grad = False

		self.optimizer = torch.optim.AdamW(
			[{'params': filter(lambda p: p.requires_grad, self.Generator.parameters()), 'initial_lr': self.config["learning_rate"]}],
			self.config["learning_rate"],betas=[self.config["adam_b1"], self.config["adam_b2"]])
		self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config["lr_decay"],
																last_epoch=-1)
		
		self.criterion = torch.nn.L1Loss()
		self.init_epoch = 0
		if self.config['resume']:
			self.resume_model(self.config['resume_path'])
		self.logging.info('config = %s', self.config)
		print('param Generator size = %fM ' % (util.count_parameters_in_M(self.Generator)))

	def make_records(self):
		time_record = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
		self.log_dir = os.path.join(self.config['out_dir'],time_record,"log")
		self.model_dir = os.path.join(self.config['out_dir'],time_record,"model")
		self.write_dir = os.path.join(self.config['out_dir'],time_record,"write")
		self.convt_mel_dir = os.path.join(self.config['out_dir'],time_record,"infer","mel")
		self.convt_voice_dir = os.path.join(self.config['out_dir'],time_record,"infer","voice")

		os.makedirs(self.log_dir, exist_ok=True)
		os.makedirs(self.model_dir, exist_ok=True)
		os.makedirs(self.write_dir, exist_ok=True)
		os.makedirs(self.convt_mel_dir, exist_ok=True)
		os.makedirs(self.convt_voice_dir, exist_ok=True)

		self.logging = util.Logger(self.log_dir, "log.txt")
		self.writer = util.Writer(self.write_dir)

	def get_test_data_loaders(self):
		test_filelist = get_test_dataset_filelist(self.config["test_wav_dir"])
		testset = Test_MelDataset(test_filelist, self.config["wav2mel_model_path"],self.config["dvector_model_path"],self.config["n_fft"],self.config["num_mels"],
							 self.config["hop_size"], self.config["win_size"], self.config["sampling_rate"],self.config["fmin"],
							 self.config["fmax"], device=self.device)
		test_data_loader = DataLoader(testset, num_workers=1, shuffle=False, sampler=None,
									  batch_size=1, pin_memory=False, drop_last=True)
		return test_data_loader

	def get_data_loaders(self, figure_mel_lable_path,batch_size):
		self.dataset = PickleDataset(figure_mel_lable_path, batch_size)
		data_loader = get_data_loader(self.dataset, batch_size=batch_size, num_workers=0, drop_last=True)
		return data_loader

	def resume_model(self, resume_path):
		checkpoint_file = resume_path
		self.logging.info('loading the model from %s' % (checkpoint_file))
		checkpoint = torch.load(checkpoint_file, map_location='cpu')
		# start epoch
		self.init_epoch = checkpoint['epoch']
		self.Generator.load_state_dict(checkpoint['Generator'])
		self.optimizer.load_state_dict(checkpoint['optimizer'])
		self.scheduler.load_state_dict(checkpoint['scheduler'])

	def reset_grad(self):
		self.optimizer.zero_grad()
	
	def train(self):
		self.Generator.train()
		for epoch in range(self.init_epoch, self.total_epochs):
			self.logging.info('************************************   train epoch %d ****************************',epoch)
			self.len_train_data = len(self.train_data_loader)
			lr = self.optimizer.state_dict()['param_groups'][0]['lr']
			self.logging.info('【train %d】lr:  %.10f', epoch, lr)
			for step, (spk_embs, input_mels,input_masks, word, overlap_lens) in enumerate(self.train_data_loader):
				spk_embs = spk_embs.cuda()
				input_mels = input_mels.cuda()
				input_masks = input_masks.cuda()
				fake_mels = self.Generator(spk_embs,input_mels,input_masks)

				losses = []
				for fake_mel, target_mel, overlap_len in zip(fake_mels.unbind(), input_mels.unbind(), overlap_lens):
					temp_loss = self.criterion(fake_mel[:overlap_len, :], target_mel[:overlap_len, :])
					losses.append(temp_loss)
				loss = sum(losses) / len(losses)

				self.reset_grad()
				loss.backward()
				self.optimizer.step()

				total_step = step + epoch * self.len_train_data
				if total_step % self.step_record_time == 0:
					self.writer.add_scalar('train/lr', lr, total_step)
					self.writer.add_scalar('train/loss' , loss, total_step)
					self.logging.info('【train_%d】 %s:  %f ', step, word[0], loss)

			if epoch % self.save_period == 0 or epoch == (self.total_epochs - 1):
				save_model_path = os.path.join(self.model_dir,'checkpoint-%d.pt' % (epoch))
				self.logging.info('saving the model to the path:%s',save_model_path)
				torch.save({'epoch': epoch + 1,
						'config': self.config,
						'Generator': self.Generator.state_dict(),
						'optimizer': self.optimizer.state_dict(),
						'scheduler': self.scheduler.state_dict()},
						save_model_path, _use_new_zipfile_serialization=False)
				# infer
				self.infer(epoch)
				self.scheduler.step()
		self.writer.close()
	
	def infer(self,epoch):
		# infer  prepare
		test_data_loader = self.get_test_data_loaders()
		# self.criterion = torch.nn.L1Loss()
		self.Generator.eval()
		mel_npy_file_list=[]
		with torch.no_grad():
			for idx, (spk_emb,input_mel, word) in enumerate(test_data_loader):
				input_mel = input_mel.cuda()
				spk_emb = spk_emb.cuda()
				fake_mel = self.Generator(spk_emb,input_mel,None)
				fake_mel = torch.clamp(fake_mel, min=0, max=1)
				fake_mel = mel_denormalize(fake_mel)
				fake_mel = fake_mel.transpose(1,2)
				fake_mel = fake_mel.detach().cpu().numpy()
				file_name = "epoch"+str(epoch)+"_"+word[0]
				mel_npy_file = os.path.join(self.convt_mel_dir, file_name+ '.npy')
				np.save(mel_npy_file, fake_mel, allow_pickle=False)
				mel_npy_file_list.append([file_name,fake_mel])
		hifi_infer(mel_npy_file_list,self.convt_voice_dir,self.config["hifi_model_path"],self.config["hifi_config_path"])
		self.Generator.train()


if __name__ == '__main__':
	print("【Solver】" )
	cudnn.benchmark = True
	config_path = r"/home/gyw/workspace/program/VC/MediumVC_G/Any2Any/config.yaml"
	with open(config_path) as f:
		config = yaml.load(f, Loader=yaml.Loader)
	solver = Solver(config)
	solver.train()
