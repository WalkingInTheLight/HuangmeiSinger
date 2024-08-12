import math

import matplotlib.pyplot as plt
import matplotlib
from torch.utils.tensorboard import SummaryWriter

from modules.fastspeech.fs2 import FastSpeech2

matplotlib.use('Agg')

import torch
import numpy as np
import os
import utils
import importlib
import glob
import torch.nn.functional as F

from tqdm import tqdm
from multiprocessing.pool import Pool

from modules.commons.ssim import ssim
from modules.fastspeech.pe import PitchExtractor
from modules.fastspeech.tts_modules import mel2ph_to_dur

from basics.base_dataset import BaseDataset
from basics.base_task import BaseTask
from basics.base_binarizer import get_pitch
from basics.base_vocoder import BaseVocoder
from modules.vocoders.registry import get_vocoder_cls
from modules.operasinger_midi.fs2 import FastSpeech2MIDI
from modules.operasinger_midi.hmxopera import FastSpeech2MIDI as DiffMIDI
from utils.hparams import hparams
from utils.plot import spec_to_figure, dur_to_figure, f0_to_figure
from utils.indexed_datasets import IndexedDataset
from utils.pitch_utils import norm_interp_f0, denorm_f0
from utils import audio



class OperaDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False):
        super().__init__(shuffle)
        self.data_dir = hparams['binary_data_dir']
        self.prefix = prefix
        self.hparams = hparams
        self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
        self.indexed_ds = None
        # self.name2spk_id={}

        # pitch stats
        f0_stats_fn = f'{self.data_dir}/train_f0s_mean_std.npy'
        if os.path.exists(f0_stats_fn):
            hparams['f0_mean'], hparams['f0_std'] = self.f0_mean, self.f0_std = np.load(f0_stats_fn)
            hparams['f0_mean'] = float(hparams['f0_mean'])
            hparams['f0_std'] = float(hparams['f0_std'])
        else:
            hparams['f0_mean'], hparams['f0_std'] = self.f0_mean, self.f0_std = None, None

        if prefix == 'test':
            if hparams['test_input_dir'] != '':
                self.indexed_ds, self.sizes = self.load_test_inputs(hparams['test_input_dir'])
            else:
                if hparams['num_test_samples'] > 0:
                    self.avail_idxs = list(range(hparams['num_test_samples'])) + hparams['test_ids']
                    self.sizes = [self.sizes[i] for i in self.avail_idxs]

    def _get_item(self, index):
        if hasattr(self, 'avail_idxs') and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
            # self.indexed_ds = IndexedDataset(self.data_dir, self.prefix)
        return self.indexed_ds[index]

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        max_frames = hparams['max_frames']
        spec = torch.Tensor(item['mel'])[:max_frames]
        energy = (spec.exp() ** 2).sum(-1).sqrt()
        mel2ph = torch.LongTensor(item['mel2ph'])[:max_frames] if 'mel2ph' in item else None
        f0, uv = norm_interp_f0(item["f0"][:max_frames], hparams)
        phone = torch.LongTensor(item['phone'][:hparams['max_input_tokens']])
        pitch = torch.LongTensor(item.get("pitch"))[:max_frames]
        # print(item.keys(), item['mel'].shape, spec.shape)
        sample = {
            "id": index,
            "item_name": item['item_name'],
            "text": item['txt'],
            "txt_token": phone,  
            "mel": spec,
            "pitch": pitch,
            "energy": energy,
            "f0": f0,
            "uv": uv,
            "mel2ph": mel2ph,
            "mel_nonpadding": spec.abs().sum(-1) > 0,
        }
        if self.hparams['use_spk_embed']:
            sample["spk_embed"] = torch.Tensor(item['spk_embed'])
        if self.hparams['use_spk_id']:
            sample["spk_id"] = item['spk_id']
            # sample['spk_id'] = 0
            # for key in self.name2spk_id.keys():
            #     if key in item['item_name']:
            #         sample['spk_id'] = self.name2spk_id[key]
            #         break

        # sample = super(OperaDataset, self).__getitem__(index)
        item = self._get_item(index)
        sample['pitch_midi'] = torch.LongTensor(item['pitch_midi'])[:hparams['max_frames']]
        sample['midi_dur'] = torch.FloatTensor(item['midi_dur'])[:hparams['max_frames']]
        sample['is_slur'] = torch.LongTensor(item['is_slur'])[:hparams['max_frames']]
        sample['word_boundary'] = torch.LongTensor(item['word_boundary'])[:hparams['max_frames']]
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        id = torch.LongTensor([s['id'] for s in samples])
        item_names = [s['item_name'] for s in samples]
        text = [s['text'] for s in samples]
        txt_tokens = utils.collate_1d([s['txt_token'] for s in samples], 0)
        f0 = utils.collate_1d([s['f0'] for s in samples], 0.0)
        pitch = utils.collate_1d([s['pitch'] for s in samples])
        uv = utils.collate_1d([s['uv'] for s in samples])
        energy = utils.collate_1d([s['energy'] for s in samples], 0.0)
        mel2ph = utils.collate_1d([s['mel2ph'] for s in samples], 0.0) \
            if samples[0]['mel2ph'] is not None else None
        mels = utils.collate_2d([s['mel'] for s in samples], 0.0)
        txt_lengths = torch.LongTensor([s['txt_token'].numel() for s in samples])
        mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples])

        batch = {
            'id': id,
            'item_name': item_names,
            'nsamples': len(samples),
            'text': text,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'mels': mels,
            'mel_lengths': mel_lengths,
            'mel2ph': mel2ph,
            'energy': energy,
            'pitch': pitch,
            'f0': f0,
            'uv': uv,
        }

        if self.hparams['use_spk_embed']:
            spk_embed = torch.stack([s['spk_embed'] for s in samples])
            batch['spk_embed'] = spk_embed
        if self.hparams['use_spk_id']:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            batch['spk_ids'] = spk_ids
        # batch = super(OperaDataset, self).collater(samples)
        batch['pitch_midi'] = utils.collate_1d([s['pitch_midi'] for s in samples], 0)
        batch['midi_dur'] = utils.collate_1d([s['midi_dur'] for s in samples], 0)
        batch['is_slur'] = utils.collate_1d([s['is_slur'] for s in samples], 0)
        batch['word_boundary'] = utils.collate_1d([s['word_boundary'] for s in samples], 0)

        return batch

    def load_test_inputs(self, test_input_dir, spk_id=0):
        inp_wav_paths = glob.glob(f'{test_input_dir}/*.wav') + glob.glob(f'{test_input_dir}/*.mp3')
        sizes = []
        items = []

        binarizer_cls = hparams.get("binarizer_cls", 'data_gen.tts.base_binarizerr.BaseBinarizer')
        pkg = ".".join(binarizer_cls.split(".")[:-1])
        cls_name = binarizer_cls.split(".")[-1]
        binarizer_cls = getattr(importlib.import_module(pkg), cls_name)
        binarization_args = hparams['binarization_args']

        for wav_fn in inp_wav_paths:
            item_name = os.path.basename(wav_fn)
            ph = txt = tg_fn = ''
            wav_fn = wav_fn
            encoder = None
            item = binarizer_cls.process_item(item_name, ph, txt, tg_fn, wav_fn, spk_id, encoder, binarization_args)
            items.append(item)
            sizes.append(item['len'])
        return items, sizes


class OperaTask(BaseTask):
    def __init__(self):
        super().__init__()

        self.phone_encoder = self.build_phone_encoder(hparams['binary_data_dir'])
        self.padding_idx = self.phone_encoder.pad()
        self.eos_idx = self.phone_encoder.eos()
        self.seg_idx = self.phone_encoder.seg()
        self.saving_result_pool = None
        self.saving_results_futures = None
        self.stats = {}

        # Initialize vocoder
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams)()

        # self.dataset_cls = MIDIDataset
        self.dataset_cls = OperaDataset
        self.sil_ph = self.phone_encoder.sil_phonemes()



    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])
        return optimizer

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=0.5)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx):
        if optimizer is None:
            return
        optimizer.step()
        optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step(self.global_step // hparams['accumulate_grad_batches'])

    def build_tts_model(self):

        if hparams.get('use_diff') is not None and hparams['use_diff']:
            self.model = DiffMIDI(self.phone_encoder)
        elif hparams.get('use_midi') is not None and hparams['use_midi']:
            self.model = FastSpeech2MIDI(self.phone_encoder)
        else:
            self.model = FastSpeech2(self.phone_encoder)

    def build_model(self):
        self.build_tts_model()
        # if hparams['load_ckpt'] != '':
        #     self.load_ckpt(hparams['load_ckpt'], strict=True)
        utils.print_arch(self.model)
        return self.model

    def run_model(self, model, sample, return_output=False, infer=False):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        target = sample['mels']  # [B, T_s, 80]
        mel2ph = sample['mel2ph']
        f0 = sample['f0']
        uv = sample['uv']
        energy = sample['energy']
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')

        output = model(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed,
                       ref_mels=target, f0=f0, uv=uv, energy=energy, infer=infer, pitch_midi=sample['pitch_midi'],
                       midi_dur=sample.get('midi_dur'), is_slur=sample.get('is_slur'))

        losses = {}

        losses['diff_f0'] = output['diff_f0']
        self.add_mel_loss(output['mel_out'], target, losses)
        self.add_dur_loss(output['dur'], mel2ph, txt_tokens, sample['word_boundary'], losses=losses)
        if hparams['use_pitch_embed']:
            self.add_pitch_loss(output, sample, losses)
        if hparams['use_energy_embed']:
            self.add_energy_loss(output['energy_pred'], energy, losses)
        if not return_output:
            return losses
        else:
            return losses, output

    def _training_step(self, sample, batch_idx, _):

        loss_output = self.run_model(self.model, sample)
        total_loss = sum([v for v in loss_output.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['txt_tokens'].size()[0]
        loss_output['lr'] = self.scheduler.get_lr()[0]
        return total_loss, loss_output

    def _validation_end(self, outputs):
        all_losses_meter = {
            'total_loss': utils.AvgrageMeter(),
        }
        for output in outputs:
            n = output['nsamples']
            for k, v in output['losses'].items():
                if k not in all_losses_meter:
                    all_losses_meter[k] = utils.AvgrageMeter()
                all_losses_meter[k].update(v, n)
            all_losses_meter['total_loss'].update(output['total_loss'], n)
        return {k: round(v.avg, 4) for k, v in all_losses_meter.items()}

    def validation_step(self, sample, batch_idx):
        outputs = {}
        txt_tokens = sample['txt_tokens']  # [B, T_t]

        target = sample['mels']  # [B, T_s, 80]
        energy = sample['energy']
        # fs2_mel = sample['fs2_mels']
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        mel2ph = sample['mel2ph']

        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(self.model, sample, return_output=True, infer=False)

        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = utils.tensors_to_scalars(outputs)


        if batch_idx < hparams['num_valid_plots']:
            model_out = self.model(
                txt_tokens, spk_embed=spk_embed, mel2ph=mel2ph, f0=None, uv=None, energy=energy, ref_mels=None,
                infer=True,
                pitch_midi=sample['pitch_midi'], midi_dur=sample.get('midi_dur'), is_slur=sample.get('is_slur'))

            if hparams.get('use_diff') is not None and hparams['use_diff']:
                gt_f0 = denorm_f0(sample['f0'], sample['uv'], hparams)
                pred_uv = model_out['uv']
                pred_f0 = denorm_f0(model_out['diff_f0'], pred_uv, hparams)
            else:
                gt_f0 = denorm_f0(sample['f0'], sample['uv'], hparams)
                pred_f0 = model_out.get('f0_denorm')

            self.plot_wav(batch_idx, sample['mels'], model_out['mel_out'], is_mel=True,
                          gt_f0=gt_f0, diff_f0=pred_f0)
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'], name=f'pred_mel')
            self.plot_f0(batch_idx, gt_f0, pred_f0, name=f'pred_f0')
        return outputs

    def test_step(self, sample, batch_idx):
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        txt_tokens = sample['txt_tokens']
        mel2ph, uv, f0 = None, None, None
        ref_mels = None
        if hparams['profile_infer']:
            pass
        else:
            if hparams['use_gt_dur']:
                mel2ph = sample['mel2ph']
            if hparams['use_gt_f0']:
                f0 = sample['f0']
                uv = sample['uv']
                print('Here using gt f0!!')
            if hparams.get('use_midi') is not None and hparams['use_midi']:
                outputs = self.model(
                    txt_tokens, spk_embed=spk_embed, mel2ph=mel2ph, f0=f0, uv=uv, ref_mels=ref_mels, infer=True,
                    pitch_midi=sample['pitch_midi'], midi_dur=sample.get('midi_dur'), is_slur=sample.get('is_slur'))
            else:
                outputs = self.model(
                    txt_tokens, spk_embed=spk_embed, mel2ph=mel2ph, f0=f0, uv=uv, ref_mels=ref_mels, infer=True)
            try:
                sample['outputs'] = self.model.out2mel(outputs['mel_out'])
            except:
                sample['outputs'] = outputs['diff_mel']

            sample['mel2ph_pred'] = outputs['mel2ph']
            if hparams.get('pe_enable') is not None and hparams['pe_enable']:
                sample['f0'] = self.pe(sample['mels'])['f0_denorm_pred']  # pe predict from GT mel
                sample['f0_pred'] = self.pe(sample['outputs'])['f0_denorm_pred']  # pe predict from Pred mel
            else:
                sample['f0'] = denorm_f0(sample['f0'], sample['uv'], hparams)
                sample['f0_pred'] = outputs.get('f0_denorm')
            return self.after_infer(sample)
    def plot_f0(self, batch_idx, f0, pre_f0, name=None):
        f0 = f0
        pred_f0 = pre_f0
        batch_idx = batch_idx
        self.logger.experiment.add_figure(f'{name}_{batch_idx}', f0_to_figure(f0[0], None, pred_f0[0]),
                                          self.global_step)


    ############
    # losses
    ############

    def add_mel_loss(self, mel_out, target, losses, bias=6.0, mel_mix_loss=None):
        if mel_mix_loss is None:
            l1_loss = F.l1_loss(mel_out, target, reduction='none')
            weights = self.weights_nonzero_speech(target)
            l1_loss = (l1_loss * weights).sum() / weights.sum()

            assert mel_out.shape == target.shape
            weights = self.weights_nonzero_speech(target)
            mel_out = mel_out[:, None] + bias
            target = target[:, None] + bias
            ssim_loss = 1 - ssim(mel_out, target, size_average=False)
            ssim_loss = (ssim_loss * weights).sum() / weights.sum()

            losses['ssim'] = hparams['mel_loss_w1'] * ssim_loss
            losses['l1'] = hparams['mel_loss_w2'] * l1_loss

        else:
            raise NotImplementedError


    def add_dur_loss(self, dur_pred, mel2ph, txt_tokens, wdb, losses=None):
        """
        :param dur_pred: [B, T], float, log scale
        :param mel2ph: [B, T]
        :param txt_tokens: [B, T]
        :param losses:
        :return:
        """
        B, T = txt_tokens.shape
        nonpadding = (txt_tokens != 0).float()
        dur_gt = mel2ph_to_dur(mel2ph, T).float() * nonpadding
        is_sil = torch.zeros_like(txt_tokens).bool()
        for p in self.sil_ph:
            is_sil = is_sil | (txt_tokens == self.phone_encoder.encode(p)[0])
        is_sil = is_sil.float()  # [B, T_txt]

        # phone duration loss
        if hparams['dur_loss'] == 'mse':
            losses['pdur'] = F.mse_loss(dur_pred, (dur_gt + 1).log(), reduction='none')
            losses['pdur'] = (losses['pdur'] * nonpadding).sum() / nonpadding.sum()
            dur_pred = (dur_pred.exp() - 1).clamp(min=0)
        else:
            raise NotImplementedError

        # use linear scale for sent and word duration
        if hparams['lambda_word_dur'] > 0:
            idx = F.pad(wdb.cumsum(axis=1), (1, 0))[:, :-1]
            # word_dur_g = dur_gt.new_zeros([B, idx.max() + 1]).scatter_(1, idx, midi_dur)  # midi_dur can be implied by add gt-ph_dur
            word_dur_p = dur_pred.new_zeros([B, idx.max() + 1]).scatter_add(1, idx, dur_pred)
            word_dur_g = dur_gt.new_zeros([B, idx.max() + 1]).scatter_add(1, idx, dur_gt)
            wdur_loss = F.mse_loss((word_dur_p + 1).log(), (word_dur_g + 1).log(), reduction='none')
            word_nonpadding = (word_dur_g > 0).float()
            wdur_loss = (wdur_loss * word_nonpadding).sum() / word_nonpadding.sum()
            losses['wdur'] = wdur_loss * hparams['lambda_word_dur']
        if hparams['lambda_sent_dur'] > 0:
            sent_dur_p = dur_pred.sum(-1)
            sent_dur_g = dur_gt.sum(-1)
            sdur_loss = F.mse_loss((sent_dur_p + 1).log(), (sent_dur_g + 1).log(), reduction='mean')
            losses['sdur'] = sdur_loss.mean() * hparams['lambda_sent_dur']

    ## dropout
    def add_diff_loss(self, x_recon, noise, losses, nonpadding=None):
        if self.loss_type == 'l1':
            if nonpadding is not None:
                loss = ((noise - x_recon).abs() * nonpadding.unsqueeze(1)).mean()
            else:
                # print('are you sure w/o nonpadding?')
                loss = (noise - x_recon).abs().mean()

        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        losses['diff_f0'] = loss

    def add_pitch_loss(self, output, sample, losses):
        mel2ph = sample['mel2ph']  # [B, T_s]
        f0 = sample['f0']
        uv = sample['uv']
        nonpadding = (mel2ph != 0).float()
        if hparams['pitch_type'] == 'frame':
            self.add_f0_loss(output['pitch_pred'], f0, uv, losses, nonpadding=nonpadding)

    def add_f0_loss(self, p_pred, f0, uv, losses, nonpadding):
        assert p_pred[..., 0].shape == f0.shape
        if hparams['use_uv']:
            assert p_pred[..., 1].shape == uv.shape
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                p_pred[:, :, 1], uv, reduction='none') * nonpadding).sum() \
                           / nonpadding.sum() * hparams['lambda_uv']
            nonpadding = nonpadding * (uv == 0).float()

        f0_pred = p_pred[:, :, 0]
        if hparams['pitch_loss'] in ['l1', 'l2']:
            pitch_loss_fn = F.l1_loss if hparams['pitch_loss'] == 'l1' else F.mse_loss
            losses['f0'] = (pitch_loss_fn(f0_pred, f0, reduction='none') * nonpadding).sum() \
                           / nonpadding.sum() * hparams['lambda_f0']
        elif hparams['pitch_loss'] == 'ssim':
            return NotImplementedError


    def add_energy_loss(self, energy_pred, energy, losses):
        nonpadding = (energy != 0).float()
        loss = (F.mse_loss(energy_pred, energy, reduction='none') * nonpadding).sum() / nonpadding.sum()
        loss = loss * hparams['lambda_energy']
        losses['e'] = loss


    def add_uv_loss(self, p_pred, uv, nonpadding, losses):
        assert p_pred[..., 1].shape == uv.shape
        losses['uv'] = (F.binary_cross_entropy_with_logits(
            p_pred[:, :, 1], uv, reduction='none') * nonpadding).sum() \
                       / nonpadding.sum() * hparams['lambda_uv']
        # nonpadding = nonpadding * (uv == 0).float()



    ############
    # validation plots
    ############
    def plot_mel(self, batch_idx, spec, spec_out, name=None):
        spec_cat = torch.cat([spec, spec_out], -1)
        # name = f'mel_{batch_idx}' if name is None else name
        vmin = hparams['mel_vmin']
        vmax = hparams['mel_vmax']
        self.logger.experiment.add_figure(f'{name}_{batch_idx}', spec_to_figure(spec_cat[0], vmin, vmax),
                                          self.global_step)


    def plot_dur(self, batch_idx, sample, model_out):
        T_txt = sample['txt_tokens'].shape[1]
        dur_gt = mel2ph_to_dur(sample['mel2ph'], T_txt)[0]
        dur_pred = self.model.dur_predictor.out2dur(model_out['dur']).float()
        txt = self.phone_encoder.decode(sample['txt_tokens'][0].cpu().numpy())
        txt = txt.split(" ")
        self.logger.experiment.add_figure(
            f'dur_{batch_idx}', dur_to_figure(dur_gt, dur_pred, txt), self.global_step)

    def plot_pitch(self, batch_idx, sample, model_out):
        f0 = sample['f0']
        f0 = denorm_f0(f0, sample['uv'], hparams)
        # f0
        uv_pred = model_out['pitch_pred'][:, :, 1] > 0
        pitch_pred = denorm_f0(model_out['pitch_pred'][:, :, 0], uv_pred, hparams)
        self.logger.experiment.add_figure(
            f'f0_{batch_idx}', f0_to_figure(f0[0], None, pitch_pred[0]), self.global_step)


    ############
    # validation plots
    ############
    def plot_wav(self, batch_idx, gt_mel, diff_mel,is_mel=False, gt_f0=None, diff_f0=None,
                 name=None):
        gt_mel = gt_mel[0].cpu().numpy()
        diff_mel = diff_mel[0].cpu().numpy()

        gt_f0 = gt_f0[0].cpu().numpy()
        diff_f0 = diff_f0[0].cpu().numpy() if diff_f0 is not None else None
        if is_mel:
            gt_wav = self.vocoder.spec2wav(gt_mel, f0=gt_f0)
            diff_wav = self.vocoder.spec2wav(diff_mel, f0=diff_f0)
        self.logger.experiment.add_audio(f'gt_{batch_idx}', gt_wav, sample_rate=hparams['audio_sample_rate'],
                                         global_step=self.global_step)
        self.logger.experiment.add_audio(f'diff_{batch_idx}', diff_wav, sample_rate=hparams['audio_sample_rate'],
                                         global_step=self.global_step)

    ############
    # infer
    ############


    def after_infer(self, predictions):
        if self.saving_result_pool is None and not hparams['profile_infer']:
            self.saving_result_pool = Pool(min(int(os.getenv('N_PROC', os.cpu_count())), 16))
            self.saving_results_futures = []
        predictions = utils.unpack_dict_to_list(predictions)
        t = tqdm(predictions)
        for num_predictions, prediction in enumerate(t):
            for k, v in prediction.items():
                if type(v) is torch.Tensor:
                    prediction[k] = v.cpu().numpy()

            item_name = prediction.get('item_name')
            text = prediction.get('text').replace(":", "%3A")[:80]

            # remove paddings
            mel_gt = prediction["mels"]
            mel_gt_mask = np.abs(mel_gt).sum(-1) > 0
            mel_gt = mel_gt[mel_gt_mask]
            mel2ph_gt = prediction.get("mel2ph")
            mel2ph_gt = mel2ph_gt[mel_gt_mask] if mel2ph_gt is not None else None
            mel_pred = prediction["outputs"]
            mel_pred_mask = np.abs(mel_pred).sum(-1) > 0
            mel_pred = mel_pred[mel_pred_mask]
            mel_gt = np.clip(mel_gt, hparams['mel_vmin'], hparams['mel_vmax'])
            mel_pred = np.clip(mel_pred, hparams['mel_vmin'], hparams['mel_vmax'])

            mel2ph_pred = prediction.get("mel2ph_pred")
            if mel2ph_pred is not None:
                if len(mel2ph_pred) > len(mel_pred_mask):
                    mel2ph_pred = mel2ph_pred[:len(mel_pred_mask)]
                mel2ph_pred = mel2ph_pred[mel_pred_mask]

            f0_gt = prediction.get("f0")
            f0_pred = prediction.get("f0_pred")
            if f0_pred is not None:
                f0_gt = f0_gt[mel_gt_mask]
                if len(f0_pred) > len(mel_pred_mask):
                    f0_pred = f0_pred[:len(mel_pred_mask)]
                f0_pred = f0_pred[mel_pred_mask]

            str_phs = None
            if self.phone_encoder is not None and 'txt_tokens' in prediction:
                str_phs = self.phone_encoder.decode(prediction['txt_tokens'], strip_padding=True)
            gen_dir = os.path.join(hparams['work_dir'],
                                   f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')

            # print("Q_vocoder: ", self.vocoder)
            wav_pred = self.vocoder.spec2wav(mel_pred, f0=f0_pred)
            if not hparams['profile_infer']:
                os.makedirs(gen_dir, exist_ok=True)
                os.makedirs(f'{gen_dir}/wavs', exist_ok=True)
                os.makedirs(f'{gen_dir}/plot', exist_ok=True)
                os.makedirs(os.path.join(hparams['work_dir'], 'P_mels_npy'), exist_ok=True)
                os.makedirs(os.path.join(hparams['work_dir'], 'G_mels_npy'), exist_ok=True)
                self.saving_results_futures.append(
                    self.saving_result_pool.apply_async(self.save_result, args=[
                        wav_pred, mel_pred, 'P', item_name, text, gen_dir, str_phs, mel2ph_pred, f0_gt, f0_pred]))

                if mel_gt is not None and hparams['save_gt']:
                    wav_gt = self.vocoder.spec2wav(mel_gt, f0=f0_gt)
                    self.saving_results_futures.append(
                        self.saving_result_pool.apply_async(self.save_result, args=[
                            wav_gt, mel_gt, 'G', item_name, text, gen_dir, str_phs, mel2ph_gt, f0_gt, f0_pred]))
                    if hparams['save_f0']:
                        import matplotlib.pyplot as plt
                        # f0_pred_, _ = get_pitch(wav_pred, mel_pred, hparams)
                        f0_pred_ = f0_pred
                        f0_gt_, _ = get_pitch(wav_gt, mel_gt, hparams)
                        fig = plt.figure()
                        plt.plot(f0_pred_, label=r'$f0_P$')
                        plt.plot(f0_gt_, label=r'$f0_G$')
                        if hparams.get('pe_enable') is not None and hparams['pe_enable']:
                            # f0_midi = prediction.get("f0_midi")
                            # f0_midi = f0_midi[mel_gt_mask]
                            # plt.plot(f0_midi, label=r'$f0_M$')
                            pass
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(f'{gen_dir}/plot/[F0][{item_name}]{text}.png', format='png')
                        plt.close(fig)

                t.set_description(
                    f"Pred_shape: {mel_pred.shape}, gt_shape: {mel_gt.shape}")
            else:
                if 'gen_wav_time' not in self.stats:
                    self.stats['gen_wav_time'] = 0
                self.stats['gen_wav_time'] += len(wav_pred) / hparams['audio_sample_rate']
                print('gen_wav_time: ', self.stats['gen_wav_time'])

        return {}

    @staticmethod
    def save_result(wav_out, mel, prefix, item_name, text, gen_dir, str_phs=None, mel2ph=None, gt_f0=None,
                    pred_f0=None):
        item_name = item_name.replace('/', '-')
        base_fn = f'[{item_name}][{prefix}]'

        if text is not None:
            base_fn += text
        base_fn += ('-' + hparams['exp_name'])
        np.save(os.path.join(hparams['work_dir'], f'{prefix}_mels_npy', item_name), mel)
        audio.save_wav(wav_out, f'{gen_dir}/wavs/{base_fn}.wav', hparams['audio_sample_rate'],
                       norm=hparams['out_wav_norm'])
        fig = plt.figure(figsize=(14, 10))
        spec_vmin = hparams['mel_vmin']
        spec_vmax = hparams['mel_vmax']
        heatmap = plt.pcolor(mel.T, vmin=spec_vmin, vmax=spec_vmax)
        fig.colorbar(heatmap)
        if hparams.get('pe_enable') is not None and hparams['pe_enable']:
            gt_f0 = (gt_f0 - 100) / (800 - 100) * 80 * (gt_f0 > 0)
            pred_f0 = (pred_f0 - 100) / (800 - 100) * 80 * (pred_f0 > 0)
            plt.plot(pred_f0, c='white', linewidth=1, alpha=0.6)
            plt.plot(gt_f0, c='red', linewidth=1, alpha=0.6)
        else:
            f0, _ = get_pitch(wav_out, mel, hparams)
            f0 = (f0 - 100) / (800 - 100) * 80 * (f0 > 0)
            plt.plot(f0, c='white', linewidth=1, alpha=0.6)
        if mel2ph is not None and str_phs is not None:
            decoded_txt = str_phs.split(" ")
            dur = mel2ph_to_dur(torch.LongTensor(mel2ph)[None, :], len(decoded_txt))[0].numpy()
            dur = [0] + list(np.cumsum(dur))
            for i in range(len(dur) - 1):
                shift = (i % 20) + 1
                plt.text(dur[i], shift, decoded_txt[i])
                plt.hlines(shift, dur[i], dur[i + 1], colors='b' if decoded_txt[i] != '|' else 'black')
                plt.vlines(dur[i], 0, 5, colors='b' if decoded_txt[i] != '|' else 'black',
                           alpha=1, linewidth=1)
        plt.tight_layout()
        plt.savefig(f'{gen_dir}/plot/{base_fn}.png', format='png', dpi=1000)
        plt.close(fig)

    ##############
    # utils
    ##############
    @staticmethod
    def expand_f0_ph(f0, mel2ph):
        f0 = denorm_f0(f0, None, hparams)
        f0 = F.pad(f0, [1, 0])
        f0 = torch.gather(f0, 1, mel2ph)  # [B, T_mel]
        return f0
