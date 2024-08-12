

import os
import pathlib
import random
from copy import deepcopy
import logging
import json
import traceback
import numpy as np
import librosa

from utils.hparams import hparams
from utils.binarizer_utils import build_phone_encoder, get_pitch
from basics.base_binarizer import BaseBinarizer, BinarizationError
from modules.vocoders.registry import VOCODERS

ALL_SHENMU = ['zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j',
              'q', 'x', 'r', 'z', 'c', 's', 'y', 'w']
ALL_YUNMU = ['a', 'ai', 'an', 'ang', 'ao', 'e', 'ei', 'en', 'eng', 'er', 'i', 'ia', 'ian',
             'iang', 'iao', 'ie', 'in', 'ing', 'iong', 'iu', 'ng', 'o', 'ong', 'ou',
             'u', 'ua', 'uai', 'uan', 'uang', 'ui', 'un', 'uo', 'v', 'van', 've', 'vn']



class HuangOperaBinarizer(BaseBinarizer):
    item2midi = {}
    item2midi_dur = {}
    item2is_slur = {}
    item2ph_durs = {}
    item2wdb = {}

    def __init__(self, processed_data_dir=None):
        if processed_data_dir is None:
            processed_data_dir = hparams['processed_data_dir']   # processed_data_dir:data/processed/popcs
        self.processed_data_dirs = processed_data_dir.split(",")
        self.binarization_args = hparams['binarization_args']
        self.item2txt = {}
        self.item2ph = {}
        self.item2wavfn = {}
        self.item2f0fn = {}
        self.item2tgfn = {}
        self.item2spk = {}



    def split_train_test_set(self, item_names):
        item_names = deepcopy(item_names)
        test_item_names = [x for x in item_names if any([x.startswith(ts) for ts in hparams['test_prefixes']])]
        train_item_names = [x for x in item_names if x not in set(test_item_names)]
        logging.info("train {}".format(len(train_item_names)))
        logging.info("test {}".format(len(test_item_names)))
        return train_item_names, test_item_names

    @property
    def train_item_names(self):
        return self._train_item_names

    @property
    def valid_item_names(self):
        return self._test_item_names

    @property
    def test_item_names(self):
        return self._test_item_names

    def process(self):
        self.load_meta_data()
        os.makedirs(hparams['binary_data_dir'], exist_ok=True)
        self.spk_map = self.build_spk_map()
        print("| spk_map: ", self.spk_map)
        spk_map_fn = f"{hparams['binary_data_dir']}/spk_map.json"
        json.dump(self.spk_map, open(spk_map_fn, 'w'))

        self.phone_encoder = self._phone_encoder()
        self.process_data('valid')
        self.process_data('test')
        self.process_data('train')

    def _phone_encoder(self):
        ph_set_fn = f"{hparams['binary_data_dir']}/phone_set.json"
        ph_set = []
        if hparams['reset_phone_dict'] or not os.path.exists(ph_set_fn):
            for ph_sent in self.item2ph.values():
                ph_set += ph_sent.split(' ')
            ph_set = sorted(set(ph_set))
            json.dump(ph_set, open(ph_set_fn, 'w'))
            print("| ph_set_length: ", len(ph_set))
            print("| Build phone set: ", ph_set)
        else:
            ph_set = json.load(open(ph_set_fn, 'r'))
            print("| ph_set_length: ", len(ph_set))
            print("| Load phone set: ", ph_set)
        return build_phone_encoder(hparams['binary_data_dir'])

    def load_meta_data(self):
        raw_data_dir = hparams['raw_data_dir']
        # meta_midi = json.load(open(os.path.join(raw_data_dir, 'meta.json')))   # [list of dict]
        try:
            utterance_labels = open(os.path.join(raw_data_dir, 'transcriptions.txt'), encoding='utf-8').readlines()
        except:
            utterance_labels = open(os.path.join(raw_data_dir, 'transcriptions.txt'), encoding='gbk').readlines()

        for utterance_label in utterance_labels:
            song_info = utterance_label.split('|')
            spkids = str(song_info[0][:6])
            for ds_id, spkids in enumerate(hparams['spkids']):
                if spkids == str(song_info[0][:6]):
                    item_name = raw_item_name = song_info[0]
                    self.item2wavfn[item_name] = f'{raw_data_dir}/wavs/{item_name}.wav'
                    self.item2txt[item_name] = song_info[1]

                    self.item2ph[item_name] = song_info[2]
                    # self.item2wdb[item_name] = list(np.nonzero([1 if x in ALL_YUNMU + ['AP', 'SP'] else 0 for x in song_info[2].split()])[0])
                    self.item2wdb[item_name] = [1 if x in ALL_YUNMU + ['AP', 'SP'] else 0 for x in song_info[2].split()]
                    # print('item: ', item_name)
                    self.item2ph_durs[item_name] = [float(x) for x in song_info[5].split(" ")]

                    self.item2midi[item_name] = [librosa.note_to_midi(x.split("/")[0]) if x != 'rest' else 0
                                                 for x in song_info[3].split(" ")]
                    self.item2midi_dur[item_name] = [float(x) for x in song_info[4].split(" ")]
                    self.item2is_slur[item_name] = [int(x) for x in song_info[6].split(" ")]
                    self.item2spk[item_name] = hparams['speakers'][ds_id]
                    # self.item2spk[item_name] = f'ds{ds_id}_{self.item2spk[item_name]}'
                        # item_name = f'ds{ds_id}_{item_name}'
                    # print(item_name,self.item2wavfn,self.item2txt,self.item2ph,self.item2spk, self.item2spk[item_name])
                    # print(self.item2spk[item_name])
                    # self.item2spk[item_name] = re.sub(u'([^\u4e00-\u9fa5])', '', str(song_info[0]))
                    # self.item2spk[item_name] = str(song_info[0][:6])


        print('OperaSpkers: ', set(self.item2spk.values()))
        self.item_names = sorted(list(self.item2txt.keys()))
        if self.binarization_args['shuffle']:
            random.seed(1234)
            random.shuffle(self.item_names)
        self._train_item_names, self._test_item_names = self.split_train_test_set(self.item_names)

    @staticmethod
    def get_pitch(wav_fn, wav, spec, ph, res):
        wav_suffix = '.wav'
        # midi_suffix = '.mid'
        wav_dir = 'wavs'
        f0_dir = 'text_f0_align'

        item_name = os.path.splitext(os.path.basename(wav_fn))[0]
        res['pitch_midi'] = np.asarray(HuangOperaBinarizer.item2midi[item_name])

        res['midi_dur'] = np.asarray(HuangOperaBinarizer.item2midi_dur[item_name])
        res['is_slur'] = np.asarray(HuangOperaBinarizer.item2is_slur[item_name])
        res['word_boundary'] = np.asarray(HuangOperaBinarizer.item2wdb[item_name])
        assert res['pitch_midi'].shape == res['midi_dur'].shape == res['is_slur'].shape, (
        res['pitch_midi'].shape, res['midi_dur'].shape, res['is_slur'].shape)


        # gt f0.
        # import sys
        # np.set_printoptions(threshold=sys.maxsize)
        gt_f0, gt_pitch_coarse = get_pitch(wav, spec, hparams)

        if sum(gt_f0) == 0:
            raise BinarizationError("Empty **gt** f0")
        res['f0'] = gt_f0
        res['pitch'] = gt_pitch_coarse

    @staticmethod
    def get_align(ph_durs, mel, phone_encoded, res, hop_size=hparams['hop_size'],
                  audio_sample_rate=hparams['audio_sample_rate']):
        mel2ph = np.zeros([mel.shape[0]], int)
        startTime = 0

        for i_ph in range(len(ph_durs)):
            start_frame = int(startTime * audio_sample_rate / hop_size + 0.5)
            end_frame = int((startTime + ph_durs[i_ph]) * audio_sample_rate / hop_size + 0.5)
            mel2ph[start_frame:end_frame] = i_ph + 1
            startTime = startTime + ph_durs[i_ph]
        res['mel2ph'] = mel2ph


    @classmethod
    def process_item(cls, item_name, ph, txt, tg_fn, wav_fn, spk_id, encoder, binarization_args):
        if hparams['vocoder'] in VOCODERS:
            wav, mel = VOCODERS[hparams['vocoder']].wav2spec(wav_fn)
        else:
            wav, mel = VOCODERS[hparams['vocoder'].split('.')[-1]].wav2spec(wav_fn)
        res = {
            'item_name': item_name, 'txt': txt, 'ph': ph, 'mel': mel, 'wav': wav, 'wav_fn': wav_fn,
            'sec': len(wav) / hparams['audio_sample_rate'], 'len': mel.shape[0], 'spk_id': spk_id
        }
        try:
            if binarization_args['with_f0']:
                cls.get_pitch(wav_fn, wav, mel, ph, res)
            if binarization_args['with_txt']:
                try:
                    phone_encoded = res['phone'] = encoder.encode(ph)
                except:
                    traceback.print_exc()
                    raise BinarizationError(f"Empty phoneme")
                if binarization_args['with_align']:
                    cls.get_align(HuangOperaBinarizer.item2ph_durs[item_name], mel, phone_encoded, res)
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None


        return res





if __name__ == "__main__":
    HuangOperaBinarizer().process()









