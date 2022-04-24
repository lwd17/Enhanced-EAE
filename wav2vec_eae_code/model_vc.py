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



class Decoder(nn.Module):
    """Decoder module:
    """
    def __init__(self, dim_neck, dim_emb, dim_pre, num_mel=80):
        super(Decoder, self).__init__()
        self.lstm1 = nn.LSTM(768, dim_pre, 1, batch_first=True)
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
        self.linear_projection = LinearNorm(1024, num_mel)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        outputs, _ = self.lstm2(x)
        decoder_output = self.linear_projection(outputs)
        return decoder_output   

    
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






def pad_layer(inp, layer, is_2d=False):
    if type(layer.kernel_size) == tuple:
        kernel_size = layer.kernel_size[0]
    else:
        kernel_size = layer.kernel_size
    if not is_2d:
        if kernel_size % 2 == 0:
            pad = (kernel_size//2, kernel_size//2 - 1)
        else:
            pad = (kernel_size//2, kernel_size//2)
    else:
        if kernel_size % 2 == 0:
            pad = (kernel_size//2, kernel_size//2 - 1, kernel_size//2, kernel_size//2 - 1)
        else:
            pad = (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2)
    inp = F.pad(inp,
            pad=pad,
            mode='reflect')
    out = layer(inp)
    return out


class PatchDiscriminator(nn.Module):
    def __init__(self, n_class=33, ns=0.2, dp=0.1):
        super(PatchDiscriminator, self).__init__()
        self.ns = ns
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=5, stride=2)
        self.conv6 = nn.Conv2d(512, 1, kernel_size=1)
        self.drop1 = nn.Dropout2d(p=dp)
        self.drop2 = nn.Dropout2d(p=dp)
        self.drop3 = nn.Dropout2d(p=dp)
        self.drop4 = nn.Dropout2d(p=dp)
        self.drop5 = nn.Dropout2d(p=dp)
        self.ins_norm1 = nn.InstanceNorm2d(self.conv1.out_channels)
        self.ins_norm2 = nn.InstanceNorm2d(self.conv2.out_channels)
        self.ins_norm3 = nn.InstanceNorm2d(self.conv3.out_channels)
        self.ins_norm4 = nn.InstanceNorm2d(self.conv4.out_channels)
        self.ins_norm5 = nn.InstanceNorm2d(self.conv5.out_channels)

    def conv_block(self, x, conv_layer, after_layers):
        out = pad_layer(x, conv_layer, is_2d=True)
        out = F.leaky_relu(out, negative_slope=self.ns)
        for layer in after_layers:
            out = layer(out)
        return out

    def forward(self, x, classify=False):
        x = torch.unsqueeze(x, dim=1)
        out = self.conv_block(x, self.conv1, [self.ins_norm1, self.drop1])
        out = self.conv_block(out, self.conv2, [self.ins_norm2, self.drop2])
        out = self.conv_block(out, self.conv3, [self.ins_norm3, self.drop3])
        out = self.conv_block(out, self.conv4, [self.ins_norm4, self.drop4])
        out = self.conv_block(out, self.conv5, [self.ins_norm5, self.drop5])
        val = pad_layer(out, self.conv6, is_2d=True)
        val = val.view(val.size(0), -1)
        mean_val = torch.mean(val, dim=1)
        return mean_val


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, dim_neck, dim_emb, dim_pre, freq, dim_spec=80, is_train=False, lr=0.001, loss_content=True,
                 multigpu=False, lambda_gan=0.0001,
                 lambda_wavenet=0.001, args=None,
                 test_path=None,wav2vec_path=None):
        super(Generator, self).__init__()

        self.decoder = Decoder(dim_neck, 0, dim_pre, num_mel=dim_spec)
        self.postnet = Postnet(num_mel=dim_spec)
        self.freq = freq

        self.loss_content = loss_content
        self.lambda_gan = lambda_gan
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
            self.opt_decoder = torch.optim.Adam(itertools.chain(self.decoder.parameters(), self.postnet.parameters()), lr=lr)
            self.opt_vocoder = torch.optim.Adam(self.vocoder.parameters(), lr=hparams.voc_lr)
            self.vocoder_loss_func = F.cross_entropy # Only for RAW


        if multigpu:
            self.decoder = nn.DataParallel(self.decoder,device_ids=[0,1])
            self.postnet = nn.DataParallel(self.postnet,device_ids=[0,1])
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
        mel_outputs = self.decoder(feat) if not self.multigpu else self.decoder.module(feat)

        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2, 1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2, 1)
        return mel_outputs_postnet

    def optimize_parameters(self, dataloader, epochs, device, display_freq=10, save_freq=1000, save_dir="./",
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
            for i, data in enumerate(dataloader):
                speaker_org, spec, prev, wav = data
                
                loss_dict, loss_dict_discriminator, loss_dict_wavenet = \
                    self.train_step(spec.to(device), speaker_org.to(device), prev=prev.to(device), wav=wav.to(device), device=device)
                if niter % display_freq == 0:
                    print("Epoch[%d] Iter[%d] Niter[%d] %s %s %s"
                          % (epoch, i, niter, loss_dict, loss_dict_discriminator, loss_dict_wavenet))
                    writer.add_scalars('data/Loss', loss_dict,
                                       niter)
                    if loss_dict_discriminator != {}:
                        writer.add_scalars('data/discriminator', loss_dict_discriminator, niter)
                    if loss_dict_wavenet != {}:
                        writer.add_scalars('data/wavenet', loss_dict_wavenet, niter)
                if niter % save_freq == 0:
                    print("Saving and Testing...", end='\t')
                    torch.save(self.state_dict(), save_dir + '/Epoch' + str(epoch).zfill(3) + '_Iter'
                               + str(niter).zfill(8) + ".pkl")
                    if len(dataloader) >= 2 and self.test_wav is not None:
                        wav_dic, sr = self.test_fixed(device)
                        for key, wav in wav_dic.items():
                            writer.add_audio(key, wav, niter, sample_rate=sr)
                        librosa.output.write_wav(save_dir + '/Iter' + str(niter).zfill(8) +'.wav', wav_dic['A_fake_w'].astype(np.float32), hparams.sample_rate)
                    print("Done")
                    self.train()
                torch.cuda.empty_cache()  # Prevent Out of Memory
                niter += 1


    def train_step(self, x, c_org, mask=None, mask_code=None, prev=None, wav=None,
                   ret_content=False, retain_graph=False, device='cuda:0'):
        # wav2vec
        wav2vec = load_pretrained_wav2vec(self.wav2vec_path).to(device)
        #print("[INFO] Wav2Vec is loaded from", self.wav2vec_path)
        #print("The net has {} parameters in total".format(sum(x.numel() for x in wav2vec.parameters())))
        
        tfm = sox.Transformer()
        tfm.vad(location=1)
        tfm.vad(location=-1)
        src_wav=wav.cpu()
        
        src_wav = torch.FloatTensor(src_wav.numpy()).to(device)
        with torch.no_grad():
            src_feat = wav2vec.extract_features(src_wav, None)["x"]
        codes = src_feat
        mel_outputs = self.decoder(codes)

        
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2, 1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2, 1)

        loss_dict, loss_dict_discriminator, loss_dict_wavenet = {}, {}, {}
        
        loss_recon = self.criterionIdt(x, mel_outputs) 
        loss_recon0 = self.criterionIdt(x, mel_outputs_postnet)
        loss_dict['recon'], loss_dict['recon0'] = loss_recon.data.item(), loss_recon0.data.item()


        loss_gen, loss_dis, loss_vocoder = [torch.from_numpy(np.array(0))] * 3
        fake_mel = None


        if not self.multigpu:
            y_hat = self.vocoder(prev,
                                self.vocoder.pad_tensor(mel_outputs_postnet, hparams.voc_pad).transpose(1, 2))
        else:
            y_hat = self.vocoder(prev,self.vocoder.module.pad_tensor(mel_outputs_postnet, hparams.voc_pad).transpose(1, 2))
        y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
        loss_vocoder = self.vocoder_loss_func(y_hat, wav.unsqueeze(-1).to(device))
        self.opt_vocoder.zero_grad()

        Loss = loss_recon + loss_recon0 + \
               self.lambda_gan * loss_gen  + self.lambda_wavenet * loss_vocoder
        loss_dict['total'] = Loss.data.item()
        self.opt_decoder.zero_grad()
        Loss.backward(retain_graph=retain_graph)
        self.opt_decoder.step()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.vocoder.parameters(), 65504.0)
        self.opt_vocoder.step()
         
        if ret_content:
            return loss_recon, loss_recon0, Loss
        return loss_dict, loss_dict_discriminator, loss_dict_wavenet
   

