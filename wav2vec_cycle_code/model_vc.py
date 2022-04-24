from pathlib import Path
import time
import imageio
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
import sox
from FragmentVC.models import load_pretrained_wav2vec
from scipy.fftpack import dct
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from copy import deepcopy
from audioUtils.audio import wav2seg, inv_preemphasis, preemphasis
from data.Sample_dataset import pad_seq
from saveWav import mel2wav
from audioUtils.hparams import hparams
from audioUtils import audio
from vocoder.models.fatchord_version import WaveRNN
import cv2
from itertools import cycle
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
_inv_mel_basis = np.linalg.pinv(audio._build_mel_basis(hparams))
mel_basis = librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=40)

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class Encoder(nn.Module):
    '''Encoder without speaker embedding'''

    def __init__(self, dim_neck, freq):
        super(Encoder, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq

        convolutions = []
        for i in range(4):
            conv_layer = nn.Sequential(
                ConvNorm(768,128 if i==3 else 768,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(128 if i==3 else 768))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(768, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, x):
        # (B, T, n_mel)
        x = x.squeeze(1).transpose(2, 1)

        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)

        return x


class Decoder(nn.Module):
    """Decoder module:
    """
    def __init__(self, dim_neck, dim_emb, dim_pre, num_mel=80):
        super(Decoder, self).__init__()
        self.conv1 = ConvNorm(768,768,kernel_size=2,stride=1,padding=5,dilation=1,w_init_gain='relu')
        self.conv2 = ConvNorm(768,768,kernel_size=1,stride=1,padding=10,dilation=1,w_init_gain='relu')
        self.conv3 = ConvNorm(768,768,kernel_size=1,stride=1,padding=10,dilation=1,w_init_gain='relu')
        self.norm1 = nn.BatchNorm1d(768) 
        self.norm2 = nn.BatchNorm1d(768) 
        self.norm3 = nn.BatchNorm1d(768) 
        self.lstm1 = nn.LSTM(2*dim_neck, dim_pre, 1, batch_first=True)
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_pre,
                         dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)
        self.cycle_projection = LinearNorm(1024,768)
        self.linear_projection = LinearNorm(768, num_mel)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        outputs, _ = self.lstm2(x)
        cycle_output = self.cycle_projection(outputs)
        decoder_output = self.linear_projection(cycle_output)
        return cycle_output,decoder_output   

    
class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, num_mel=80):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(num_mel, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, num_mel,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(num_mel))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x








class Generator(nn.Module):
    """Generator network."""
    def __init__(self, dim_neck, dim_emb, dim_pre, freq, dim_spec=80, is_train=False, lr=0.001, loss_content=True,
                 discriminator=False, multigpu=False, lambda_gan=0.0001,
                 lambda_wavenet=0.001, args=None,
                 test_path=None,wav2vec_path=None):
        super(Generator, self).__init__()

        self.encoder = Encoder(dim_neck, freq)
        self.decoder1 = Decoder(dim_neck, 0, dim_pre, num_mel=dim_spec)
        self.decoder2 = Decoder(dim_neck, 0, dim_pre, num_mel=dim_spec)
        self.postnet1 = Postnet(num_mel=dim_spec)
        self.postnet2 = Postnet(num_mel=dim_spec)
        self.freq = freq

        self.loss_content = loss_content
        self.lambda_wavenet = lambda_wavenet
        self.wav2vec_path=wav2vec_path
        self.multigpu = multigpu
        if test_path is not None:
            self.prepare_test(dim_spec, test_path)

        self.vocoder = WaveRNN(
            rnn_dims=hparams.voc_rnn_dims,
            fc_dims=hparams.voc_fc_dims,
            bits=hparams.bits,
            pad=hparams.voc_pad,
            upsample_factors=hparams.voc_upsample_factors,
            feat_dims=hparams.num_mels,
            compute_dims=hparams.voc_compute_dims,
            res_out_dims=hparams.voc_res_out_dims,
            res_blocks=hparams.voc_res_blocks,
            hop_length=hparams.hop_size,
            sample_rate=hparams.sample_rate,
            mode=hparams.voc_mode
        )
        
        if is_train:
            self.criterionIdt = torch.nn.L1Loss(reduction='mean')
            self.opt_encoder = torch.optim.Adam(self.encoder.parameters(), lr=lr)
            self.opt_decoder1 = torch.optim.Adam(itertools.chain(self.decoder1.parameters(), self.postnet1.parameters()), lr=lr)
            self.opt_decoder2 = torch.optim.Adam(itertools.chain(self.decoder2.parameters(), self.postnet2.parameters()), lr=lr)
            self.opt_vocoder = torch.optim.Adam(self.vocoder.parameters(), lr=hparams.voc_lr)
            self.vocoder_loss_func = F.cross_entropy # Only for RAW


        if multigpu:
            self.encoder = nn.DataParallel(self.encoder,device_ids=[0,1])
            self.decoder1 = nn.DataParallel(self.decoder1,device_ids=[0,1])
            self.decoder2 = nn.DataParallel(self.decoder2,device_ids=[0,1])
            self.postnet1 = nn.DataParallel(self.postnet1,device_ids=[0,1])
            self.postnet2 = nn.DataParallel(self.postnet2,device_ids=[0,1])
            self.vocoder = nn.DataParallel(self.vocoder)

    def prepare_test(self, dim_spec, test_path):
        mel_basis80 = librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=80)

        wav, sr = librosa.load(test_path, hparams.sample_rate)
        wav = preemphasis(wav, hparams.preemphasis, hparams.preemphasize)
        linear_spec = np.abs(
            librosa.stft(wav, n_fft=hparams.n_fft, hop_length=hparams.hop_size, win_length=hparams.win_size))
        mel_spec = mel_basis80.dot(linear_spec)
        mel_db = 20 * np.log10(mel_spec)
        source_spec = np.clip((mel_db + 120) / 125, 0, 1)

        self.test_wav = wav

        self.test_spec = torch.Tensor(pad_seq(source_spec.T, hparams.freq)).unsqueeze(0)

    def test_fixed(self, device):
        with torch.no_grad():
            s2t_spec = self.conversion(self.test_wav, device).cpu()

        ret_dic = {}
        ret_dic['A_fake_griffin'], sr = mel2wav(s2t_spec.numpy().squeeze(0).T)
        ret_dic['A'] = self.test_wav

        with torch.no_grad():
            if not self.multigpu:
                ret_dic['A_fake_w'] = inv_preemphasis(self.vocoder.generate(s2t_spec.to(device).transpose(2, 1), False, None, None, mu_law=True),
                                                hparams.preemphasis, hparams.preemphasize)
            else:
                ret_dic['A_fake_w'] = inv_preemphasis(self.vocoder.module.generate(s2t_spec.to(device).transpose(2, 1), False, None, None, mu_law=True),
                                                hparams.preemphasis, hparams.preemphasize)
        return ret_dic, sr


    def conversion(self, wav, device, speed=1):
        wav2vec = load_pretrained_wav2vec(self.wav2vec_path).to(device)
        tfm = sox.Transformer()
        tfm.vad(location=1)
        tfm.vad(location=-1)
        
        src_wav = torch.FloatTensor(wav).unsqueeze(0).to(device)
        with torch.no_grad():
            src_feat = wav2vec.extract_features(src_wav, None)["x"]
        times = int(src_feat.size(1) / self.freq)
        feat = src_feat[:,:self.freq*times,:]
        latent_code = self.encoder(feat) if not self.multigpu else self.encoder.module(feat)
        _, mel_outputs = self.decoder1(latent_code) if not self.multigpu else self.decoder1.module(latent_code)

        mel_outputs_postnet = self.postnet1(mel_outputs.transpose(2, 1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2, 1)
        return mel_outputs_postnet

    def optimize_parameters(self, dataloader1,dataloader2, epochs, device, display_freq=10, save_freq=1000, save_dir="./",
                            experimentName="Train", load_model=None, initial_niter=0):
        writer = SummaryWriter(log_dir="logs/"+experimentName)
        if load_model is not None:
            print("Loading from %s..." % load_model)
            d = torch.load(load_model)
            newdict = d.copy()
            for key, value in d.items():
                newkey = key
                if 'wavenet' in key:
                    newdict[key.replace('wavenet', 'vocoder')] = newdict.pop(key)
                    newkey = key.replace('wavenet', 'vocoder')
                if self.multigpu and 'module' not in key:
                    newdict[newkey.replace('.','.module.',1)] = newdict.pop(newkey)
                    newkey = newkey.replace('.', '.module.', 1)
                if newkey not in self.state_dict():
                    newdict.pop(newkey)
            self.load_state_dict(newdict)
            print("AutoVC Model Loaded")
        niter = initial_niter
        for epoch in range(epochs):
            self.train()
            for i, data in enumerate(zip(cycle(dataloader1),dataloader2)):
                speaker_org1, spec1, prev1, wav1 = data[0]
                speaker_org2, spec2, prev2, wav2 = data[1]
                
                loss_dict, loss_dict_discriminator, loss_dict_wavenet = \
                    self.train_step(spec1.to(device), speaker_org1.to(device), spec2.to(device), speaker_org2.to(device),prev1=prev1.to(device), wav1=wav1.to(device),prev2=prev2.to(device),wav2=wav2.to(device), device=device)
                if niter % display_freq == 0:
                    print("Epoch[%d] Iter[%d] Niter[%d] %s %s %s"
                          % (epoch, i, niter, loss_dict, loss_dict_discriminator, loss_dict_wavenet))
                    writer.add_scalars('data/Loss', loss_dict,
                                       niter)
                    if loss_dict_wavenet != {}:
                        writer.add_scalars('data/wavenet', loss_dict_wavenet, niter)
                if niter % save_freq == 0:
                    print("Saving and Testing...", end='\t')
                    torch.save(self.state_dict(), save_dir + '/Epoch' + str(epoch).zfill(3) + '_Iter'
                               + str(niter).zfill(8) + ".pkl")
                    if len(dataloader1) >= 2 and self.test_wav is not None:
                        wav_dic, sr = self.test_fixed(device)
                        for key, wav in wav_dic.items():
                            writer.add_audio(key, wav, niter, sample_rate=sr)
                        librosa.output.write_wav(save_dir + '/Iter' + str(niter).zfill(8) +'.wav', wav_dic['A_fake_w'].astype(np.float32), hparams.sample_rate)
                    print("Done")
                    self.train()
                torch.cuda.empty_cache()  # Prevent Out of Memory
                niter += 1


    def train_step(self, x1, c_org1,x2, c_org2, mask=None, mask_code=None, prev1=None, wav1=None,
                   prev2=None,wav2=None,ret_content=False, retain_graph=False, device='cuda:0'):
        # encoder:
        wav2vec = load_pretrained_wav2vec(self.wav2vec_path).to(device)
        # wav2vec
        #print("[INFO] Wav2Vec is loaded from", self.wav2vec_path)
        #print("The net has {} parameters in total".format(sum(x.numel() for x in wav2vec.parameters())))
        
        tfm = sox.Transformer()
        tfm.vad(location=1)
        tfm.vad(location=-1)
        src_wav1=wav1.cpu()
        src_wav2=wav2.cpu()
        
        src_wav1 = torch.FloatTensor(src_wav1.numpy()).to(device)
        src_wav2 = torch.FloatTensor(src_wav2.numpy()).to(device)
        with torch.no_grad():
            src_feat1 = wav2vec.extract_features(src_wav1, None)["x"]
            src_feat2 = wav2vec.extract_features(src_wav2, None)["x"]
        codes1 = self.encoder(src_feat1)
        codes2 = self.encoder(src_feat2)
        cycle_output1, mel_outputs1 = self.decoder1(codes1)
        cycle_output2, mel_outputs2 = self.decoder2(codes2)
        cycle_outputs1, _ = self.decoder2(codes1)
        cycle_outputs2, _ = self.decoder1(codes2)

        
        mel_outputs_postnet1 = self.postnet1(mel_outputs1.transpose(2, 1))
        mel_outputs_postnet1 = mel_outputs1 + mel_outputs_postnet1.transpose(2, 1)
        mel_outputs_postnet2 = self.postnet2(mel_outputs2.transpose(2, 1))
        mel_outputs_postnet2 = mel_outputs2 + mel_outputs_postnet2.transpose(2, 1)

        loss_dict, loss_dict_discriminator, loss_dict_wavenet = {}, {}, {}
        
        loss_recon = self.criterionIdt(x1, mel_outputs1) + self.criterionIdt(x2, mel_outputs2)
        loss_recon0 = self.criterionIdt(x1, mel_outputs_postnet1) + self.criterionIdt(x2,mel_outputs_postnet2)
        loss_dict['recon'], loss_dict['recon0'] = loss_recon.data.item(), loss_recon0.data.item()

        if self.loss_content:
            cycle_codes1 = self.encoder(cycle_outputs1)
            cycle_codes2 = self.encoder(cycle_outputs2)
            
            loss_content = self.criterionIdt(codes1,cycle_codes1) + self.criterionIdt(codes2,cycle_codes2)
            loss_dict['content'] = loss_content.data.item()
        else:
            loss_content = torch.from_numpy(np.array(0))

        loss_gen, loss_dis, loss_vocoder = [torch.from_numpy(np.array(0))] * 3
        fake_mel = None


        if not self.multigpu:
            y_hat1 = self.vocoder(prev1,
                                self.vocoder.pad_tensor(mel_outputs_postnet1, hparams.voc_pad).transpose(1, 2))
        else:
            y_hat1 = self.vocoder(prev1,self.vocoder.module.pad_tensor(mel_outputs_postnet1, hparams.voc_pad).transpose(1, 2))
        y_hat1 = y_hat1.transpose(1, 2).unsqueeze(-1)
        loss_vocoder = self.vocoder_loss_func(y_hat1, wav1.unsqueeze(-1).to(device))
        self.opt_vocoder.zero_grad()

        Loss = loss_recon + loss_recon0 + 0.1 * loss_content + \
               + self.lambda_wavenet * loss_vocoder
        loss_dict['total'] = Loss.data.item()
        self.opt_encoder.zero_grad()
        self.opt_decoder1.zero_grad()
        self.opt_decoder2.zero_grad()
        Loss.backward(retain_graph=retain_graph)
        self.opt_encoder.step()
        self.opt_decoder1.step()
        self.opt_decoder2.step()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.vocoder.parameters(), 65504.0)
        self.opt_vocoder.step()
         
        if ret_content:
            return loss_recon, loss_recon0, loss_content, Loss, content
        return loss_dict, loss_dict_discriminator, loss_dict_wavenet
   

