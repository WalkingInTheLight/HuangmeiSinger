from modules.commons.common_layers import *
from modules.commons.common_layers import Embedding
from modules.fastspeech.tts_modules import FastspeechDecoder, DurationPredictor, LengthRegulator, PitchPredictor, \
    EnergyPredictor, FastspeechEncoder
from utils.hparams import hparams
from utils.pitch_utils import f0_to_coarse, denorm_f0, norm_f0
from modules.fastspeech.fs2 import FastSpeech2
from modules.diff.ddpm import PitchDiffusion
from modules.commons.conformer.conformer import ConformerEncoder, ConformerDecoder
from modules.commons.conformer.branchformer import BranchformerEncoder, BranchformerDecoder
from modules.commons.conformer.ebranchformer import EBranchformerEncoder, EBranchformerDecoder

class ConformerMIDIEncoder(ConformerEncoder):

    def forward(self, x, midi_embedding, midi_dur_embedding, slur_embedding):
        """

        :param src_tokens: [B, T]
        :return: [B x T x C]
        """
        x = self.embed(x)  # [B, T, H]
        x = x + midi_embedding + midi_dur_embedding + slur_embedding
        x = super(ConformerEncoder, self).forward(x)
        return x



class BranchformerMIDIEncoder(BranchformerEncoder):

    def forward(self, x, midi_embedding, midi_dur_embedding, slur_embedding):
        """

        :param src_tokens: [B, T]
        :return: [B x T x C]
        """
        x = self.embed(x)  # [B, T, H]
        x = x + midi_embedding + midi_dur_embedding + slur_embedding
        x = super(BranchformerEncoder, self).forward(x)
        return x


class EBranchformerMIDIEncoder(EBranchformerEncoder):

    def forward(self, x, midi_embedding, midi_dur_embedding, slur_embedding):
        """

        :param src_tokens: [B, T]
        :return: [B x T x C]
        """
        x = self.embed(x)  # [B, T, H]
        x = x + midi_embedding + midi_dur_embedding + slur_embedding
        x = super(EBranchformerEncoder, self).forward(x)
        return x


class FastspeechMIDIEncoder(FastspeechEncoder):
    def forward_embedding(self, txt_tokens, midi_embedding, midi_dur_embedding, slur_embedding):
        x = self.embed_scale * self.embed_tokens(txt_tokens)
        x = x + midi_embedding + midi_dur_embedding + slur_embedding
        if hparams['use_pos_embed']:
            if hparams.get('rel_pos') is not None and hparams['rel_pos']:
                x = self.embed_positions(x)
            else:
                positions = self.embed_positions(txt_tokens)
                x = x + positions
        x = F.dropout(x, p=self.dropout, training=self.training)  
        return x

    def forward(self, txt_tokens, midi_embedding, midi_dur_embedding, slur_embedding):
        """

        :param txt_tokens: [B, T]
        :return: {
            'encoder_out': [T x B x C]
        }
        """
        encoder_padding_mask = txt_tokens.eq(self.padding_idx).data
        x = self.forward_embedding(txt_tokens, midi_embedding, midi_dur_embedding, slur_embedding)  # [B, T, H]
        x = super(FastspeechEncoder, self).forward(x, encoder_padding_mask)
        return x



FS_ENCODERS = {
    'fft': lambda hp, embed_tokens, d: FastspeechMIDIEncoder(
        embed_tokens, hp['hidden_size'], hp['enc_layers'], hp['enc_ffn_kernel_size'],
        num_heads=hp['num_heads']),

    'conformer': lambda hp, embed_tokens, d: ConformerMIDIEncoder(
        embed_tokens, hp['hidden_size'], hp['enc_layers'], hp['enc_ffn_kernel_size'],
        num_heads=hp['num_heads']),

    'branchformer': lambda hp, embed_tokens, d: BranchformerMIDIEncoder(
        embed_tokens, hp['hidden_size'], hp['enc_layers'], hp['enc_ffn_kernel_size'],
        num_heads=hp['num_heads']),

    'ebranchformer': lambda hp, embed_tokens, d: EBranchformerMIDIEncoder(
        embed_tokens, hp['hidden_size'], hp['enc_layers'], hp['enc_ffn_kernel_size'],
        num_heads=hp['num_heads']),
}


class FastSpeech2MIDI(FastSpeech2):
    def __init__(self, dictionary, out_dims=None):
        super().__init__(dictionary, out_dims)
        del self.encoder
        self.encoder = FS_ENCODERS[hparams['encoder_type']](hparams, self.encoder_embed_tokens, self.dictionary)
        self.midi_embed = Embedding(300, self.hidden_size, self.padding_idx)
        self.midi_dur_layer = Linear(1, self.hidden_size)
        self.is_slur_embed = Embedding(2, self.hidden_size)


        # 音高扩散
        if hparams.get('use_diff') is not None and hparams['use_diff']:
            pitch_hparams = hparams['pitch_prediction_args']
            self.pitch_diffusion = PitchDiffusion(
                vmin=pitch_hparams['pitd_norm_min'],
                vmax=pitch_hparams['pitd_norm_max'],
                cmin=pitch_hparams['pitd_clip_min'],
                cmax=pitch_hparams['pitd_clip_max'],
                repeat_bins=pitch_hparams['repeat_bins'],
                timesteps=hparams['timesteps'],
                k_step=hparams['K_step'],
                loss_type=hparams['diff_loss_type'],
                denoiser_type=hparams['diff_decoder_type'],
                denoiser_args={
                    'n_layers': pitch_hparams['residual_layers'],
                    'n_chans': pitch_hparams['residual_channels'],
                    'n_dilates': pitch_hparams['dilation_cycle_length'],
                }
            )

    def forward(self, txt_tokens, mel2ph=None, spk_embed=None,
                ref_mels=None, f0=None, uv=None, energy=None, skip_decoder=False,
                spk_embed_dur_id=None, spk_embed_f0_id=None, infer=False, **kwargs):
        ret = {}

        midi_embedding = self.midi_embed(kwargs['pitch_midi'])
        midi_dur_embedding, slur_embedding = 0, 0
        if kwargs.get('midi_dur') is not None:
            midi_dur_embedding = self.midi_dur_layer(kwargs['midi_dur'][:, :, None])  # [B, T, 1] -> [B, T, H]
        if kwargs.get('is_slur') is not None:
            slur_embedding = self.is_slur_embed(kwargs['is_slur'])
        encoder_out = self.encoder(txt_tokens, midi_embedding, midi_dur_embedding, slur_embedding)  # [B, T, C]
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]

        # add ref style embed
        # Not implemented
        # variance encoder
        var_embed = 0

        # encoder_out_dur denotes encoder outputs for duration predictor
        # in speech adaptation, duration predictor use old speaker embedding
        if hparams['use_spk_embed']:
            spk_embed_dur = spk_embed_f0 = spk_embed = self.spk_embed_proj(spk_embed)[:, None, :]
        elif hparams['use_spk_id']:
            spk_embed_id = spk_embed
            if spk_embed_dur_id is None:
                spk_embed_dur_id = spk_embed_id
            if spk_embed_f0_id is None:
                spk_embed_f0_id = spk_embed_id
            spk_embed = self.spk_embed_proj(spk_embed_id)[:, None, :]
            spk_embed_dur = spk_embed_f0 = spk_embed
            if hparams['use_split_spk_id']:
                spk_embed_dur = self.spk_embed_dur(spk_embed_dur_id)[:, None, :]
                spk_embed_f0 = self.spk_embed_f0(spk_embed_f0_id)[:, None, :]
        else:
            spk_embed_dur = spk_embed_f0 = spk_embed = 0


        # add dur
        dur_inp = (encoder_out + var_embed + spk_embed_dur) * src_nonpadding

        mel2ph = self.add_dur(dur_inp, mel2ph, txt_tokens, ret)

        decoder_inp = F.pad(encoder_out, [0, 0, 1, 0])

        mel2ph_ = mel2ph[..., None].repeat([1, 1, encoder_out.shape[-1]])
        decoder_inp_origin = decoder_inp = torch.gather(decoder_inp, 1, mel2ph_)  # [B, T, H]

        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]

        # add pitch and energy embed
        pitch_inp = (decoder_inp_origin + var_embed + spk_embed_f0) * tgt_nonpadding
        diff_cond = decoder_inp

        if hparams.get('use_diff') is not None and hparams['use_diff']:
            # 音高扩散模型
            if infer:
                pitch_pred_out = self.pitch_diffusion(pitch_inp, infer=True)
            else:
                pitch_pred_out = self.pitch_diffusion(pitch_inp, f0, infer=False)

        if hparams['use_pitch_embed']:
            pitch_inp_ph = (encoder_out + var_embed + spk_embed_f0) * src_nonpadding
            decoder_inp = decoder_inp + self.add_pitch(pitch_inp, f0, uv, mel2ph, ret, encoder_out=pitch_inp_ph)


        if hparams.get('use_diff') is not None and hparams['use_diff']:
            pitch_padding = mel2ph == 0
            if infer or f0 is None:
                ret['decoder_inp'] = pitch_inp
                f0 = pitch_pred_out
                uv = ret['pitch_pred'][:, :, 1] > 0
            else:
                f0 = f0
                uv = uv

            f0_denorm = denorm_f0(f0, uv, hparams, pitch_padding=pitch_padding)
            pitch = f0_to_coarse(f0_denorm)  # start from 0
            pitch_embed = self.pitch_embed(pitch)
            diff_cond = diff_cond + pitch_embed
            if hparams['use_energy_embed']:
                diff_cond = diff_cond + self.add_energy(pitch_inp, energy, ret)
            diff_cond = (diff_cond + spk_embed) * tgt_nonpadding

            ret['diff_decoder_cond'] = diff_cond
            ret['uv'] = uv
            ret['diff_f0'] = pitch_pred_out
            # if skip_decoder:
            #     return ret
            #
            ret['mel_out'] = self.run_decoder(diff_cond, tgt_nonpadding, ret, infer=infer, **kwargs)

        else:
            if hparams['use_energy_embed']:
                decoder_inp = decoder_inp + self.add_energy(pitch_inp, energy, ret)
            ret['fs2_decoder_inp'] = decoder_inp = (decoder_inp + spk_embed) * tgt_nonpadding
            # if skip_decoder:
            #     return ret
            ret['mel_out'] = self.run_decoder(decoder_inp, tgt_nonpadding, ret, infer=infer, **kwargs)

        return ret
