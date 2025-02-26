from tasks.tts.dataset_utils import FastSpeechDataset
import torch
from utils.commons.dataset_utils import collate_1d_or_2d
from tasks.tts.dataset_utils import BaseSpeechDataset
from utils.audio.pitch.utils import norm_interp_f0, denorm_f0
import random
from utils.commons.indexed_datasets import IndexedDataset
from tqdm import tqdm
import numpy as np

class TechDataset(FastSpeechDataset):
    def __getitem__(self, index):
        hparams=self.hparams
        sample = super(TechDataset, self).__getitem__(index)
        item = self._get_item(index)
        note = torch.LongTensor(item['ep_pitches'][:hparams['max_input_tokens']])
        note_dur = torch.FloatTensor(item['ep_notedurs'][:hparams['max_input_tokens']])
        note_type = torch.LongTensor(item['ep_types'][:hparams['max_input_tokens']])
        sample["note"], sample["note_dur"], sample["note_type"] = note, note_dur, note_type

        for key in ['mix_tech','falsetto_tech','breathy_tech','bubble_tech','strong_tech','weak_tech','pharyngeal_tech','vibrato_tech','glissando_tech']:
            if key not in item:
                item[key] = [2] * len(item['ph'])

        mix = torch.LongTensor(item['mix_tech'][:hparams['max_input_tokens']])
        falsetto= torch.LongTensor(item['falsetto_tech'][:hparams['max_input_tokens']])
        breathy = torch.LongTensor(item['breathy_tech'][:hparams['max_input_tokens']])
        sample['mix'],sample['falsetto'],sample['breathy']=mix,falsetto,breathy

        bubble = torch.LongTensor(item['bubble_tech'][:hparams['max_input_tokens']])
        strong = torch.LongTensor(item['strong_tech'][:hparams['max_input_tokens']])
        weak = torch.LongTensor(item['weak_tech'][:hparams['max_input_tokens']])
        sample['bubble'],sample['strong'],sample['weak']=bubble,strong,weak

        pharyngeal = torch.LongTensor(item['pharyngeal_tech'][:hparams['max_input_tokens']])
        vibrato = torch.LongTensor(item['vibrato_tech'][:hparams['max_input_tokens']])
        glissando = torch.LongTensor(item['glissando_tech'][:hparams['max_input_tokens']])
        sample['pharyngeal'],sample['vibrato'],sample['glissando'] = pharyngeal,vibrato,glissando
        
        return sample
    
    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super(TechDataset, self).collater(samples)
        notes = collate_1d_or_2d([s['note'] for s in samples], 0.0)
        note_durs = collate_1d_or_2d([s['note_dur'] for s in samples], 0.0)
        note_types = collate_1d_or_2d([s['note_type'] for s in samples], 0.0)
        batch["notes"], batch["note_durs"], batch["note_types"] = notes, note_durs, note_types

        mix = collate_1d_or_2d([s['mix'] for s in samples], 0.0)
        falsetto = collate_1d_or_2d([s['falsetto'] for s in samples], 0.0)
        breathy = collate_1d_or_2d([s['breathy'] for s in samples], 0.0)
        batch['mix'],batch['falsetto'],batch['breathy']=mix,falsetto,breathy

        bubble = collate_1d_or_2d([s['bubble'] for s in samples], 0.0)
        strong = collate_1d_or_2d([s['strong'] for s in samples], 0.0)
        weak = collate_1d_or_2d([s['weak'] for s in samples], 0.0)
        batch['bubble'],batch['strong'],batch['weak']=bubble,strong,weak

        pharyngeal = collate_1d_or_2d([s['pharyngeal'] for s in samples], 0.0)
        vibrato = collate_1d_or_2d([s['vibrato'] for s in samples], 0.0)
        glissando = collate_1d_or_2d([s['glissando'] for s in samples], 0.0)
        batch['pharyngeal'],batch['vibrato'],batch['glissando'] = pharyngeal,vibrato,glissando    

        return batch

