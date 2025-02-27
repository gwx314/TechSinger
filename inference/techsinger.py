import os
import numpy as np
import torch
from inference.tts.base_tts_infer import BaseTTSInfer
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams
from utils.text.text_encoder import build_token_encoder
from modules.TCSinger.tcsinger import SADecoder
from modules.TCSinger.sdlm import SDLM
from tasks.tts.vocoder_infer.base_vocoder import get_vocoder_cls
from utils.audio import librosa_wav2spec
from utils.commons.hparams import set_hparams
from utils.commons.hparams import hparams as hp
from utils.audio.io import save_wav
import json
from modules.TechSinger.techsinger import RFSinger, RFPostnet

def process_align(ph_durs, mel, item, hop_size ,audio_sample_rate):
    mel2ph = np.zeros([mel.shape[0]], int)
    startTime = 0

    for i_ph in range(len(ph_durs)):
        start_frame = int(startTime * audio_sample_rate / hop_size + 0.5)
        end_frame = int((startTime + ph_durs[i_ph]) * audio_sample_rate / hop_size + 0.5)
        mel2ph[start_frame:end_frame] = i_ph + 1
        startTime = startTime + ph_durs[i_ph]

    return mel2ph

def self_clone(x):
    if x == None:
        return None
    y = x.clone()
    result = torch.cat((x, y), dim=0)
    return result

class techinfer(BaseTTSInfer):
    def build_model(self):
        dict_size = len(self.ph_encoder)
        model = RFSinger(dict_size, self.hparams)
        model.eval()
        load_ckpt(model, hparams['fs2_ckpt_dir'], strict=True)     
        self.model_post=RFPostnet()
        
        load_ckpt(self.model_post, os.path.join('checkpoints', hparams['exp_name']), strict=True)
        self.model_post.eval()
        self.model_post.to(self.device)

        binary_data_dir = hparams['binary_data_dir']
        self.ph_encoder = build_token_encoder(f'{binary_data_dir}/phone_set.json')
        return model

    def build_vocoder(self):
        vocoder = get_vocoder_cls(hparams["vocoder"])()
        return vocoder

    def forward_model(self, inp):
        sample = self.input_to_batch(inp)

        txt_tokens = sample['txt_tokens']  # [B, T_t]
        txt_lengths = sample['txt_lengths']
        notes, note_durs,note_types = sample["notes"], sample["note_durs"],sample['note_types']
        mel2ph=None
        spk_id=sample['spk_id']
        mix,falsetto,breathy=sample['mix'],sample['falsetto'],sample['breathy']
        pharyngeal,vibrato,glissando = sample['pharyngeal'],sample['vibrato'],sample['glissando']
        bubble,strong,weak = sample['bubble'],sample['strong'],sample['weak']
        f0 = uv = None
        output = {}
        
        # Run model
        with torch.no_grad():
            umix, ufalsetto, ubreathy = torch.ones_like(mix, dtype=mix.dtype) * 2, torch.ones_like(falsetto, dtype=falsetto.dtype) * 2, torch.ones_like(breathy, dtype=breathy.dtype) * 2
            ububble, ustrong, uweak = torch.ones_like(mix, dtype=mix.dtype) * 2, torch.ones_like(strong, dtype=strong.dtype) * 2, torch.ones_like(weak, dtype=weak.dtype) * 2
            upharyngeal, uvibrato, uglissando = torch.ones_like(bubble, dtype=bubble.dtype) * 2, torch.ones_like(vibrato, dtype=vibrato.dtype) * 2, torch.ones_like(glissando, dtype=glissando.dtype) * 2
            mix = torch.cat((mix, umix), dim=0)
            falsetto = torch.cat((falsetto, ufalsetto), dim=0)
            breathy = torch.cat((breathy, ubreathy), dim=0)
            bubble = torch.cat((bubble, ububble), dim=0)
            strong = torch.cat((strong, ustrong), dim=0)
            weak = torch.cat((weak, uweak), dim=0)
            pharyngeal = torch.cat((pharyngeal, upharyngeal), dim=0)
            vibrato = torch.cat((vibrato, uvibrato), dim=0)
            glissando = torch.cat((glissando, uglissando), dim=0)
            
            txt_tokens = self_clone(txt_tokens)
            mel2ph = self_clone(mel2ph)
            spk_id = self_clone(spk_id)
            f0 = self_clone(f0)
            uv = self_clone(uv)
            notes = self_clone(notes)
            note_durs = self_clone(note_durs)
            note_types = self_clone(note_types)
            
            output = self.model(txt_tokens, mel2ph=mel2ph, spk_id=spk_id, f0=f0, uv=uv, 
                                note=notes, note_dur=note_durs, note_type=note_types,
                                mix=mix, falsetto=falsetto, breathy=breathy,
                                bubble=bubble, strong=strong, weak=weak,
                                pharyngeal=pharyngeal, vibrato=vibrato, glissando=glissando, 
                                infer=True)
            self.model_post(None, True, output, True, cfg_scale=1.0,  noise=None)
            mel_out =  output['mel_out'][0]
            pred_f0 = output.get('f0_denorm_pred')[0]
            wav_out = self.vocoder.spec2wav(mel_out.cpu(),f0=pred_f0.cpu())

        return wav_out, mel_out
    

    def preprocess_input(self, inp):
        """
        :param inp: {'text': str, 'item_name': (str, optional), 'spk_name': (str, optional)}
        :return:
        """
        ph_gen=' '.join(inp['text_gen'])
        ph_token = self.ph_encoder.encode(ph_gen)
        note=inp['note_gen']
        note_dur=inp['note_dur_gen']
        note_type=inp['note_type_gen']
        tech_list=inp['tech_list']

        mix=[]
        falsetto=[]
        breathy=[]
        pharyngeal=[]
        vibrato=[]
        glissando=[]
        bubble=[]
        strong=[]
        weak=[]
        for element in tech_list:
            mix.append(1 if '1' in element else 0)
            falsetto.append(1 if '2' in element else 0)
            breathy.append(1 if '3' in element else 0)
            pharyngeal.append(1 if '4' in element else 0)
            vibrato.append(1 if '5' in element else 0)
            glissando.append(1 if '6' in element else 0)
            bubble.append(1 if '7' in element else 0)
            strong.append(1 if '8' in element else 0)
            weak.append(1 if '9' in element else 0)

        item = {'item_name': inp['gen'], 'text': inp['text_gen'], 'ph': inp['text_gen'],
                'ph_token': ph_token, 'spk_id':inp['spk_id'],
                'mel2ph': None, 'note':note, 'note_dur':note_dur,'note_type':note_type,
                'mix_tech': mix, 'falsetto_tech': falsetto, 'breathy_tech': breathy,
                'pharyngeal_tech':pharyngeal , 'vibrato_tech':vibrato,'glissando_tech':glissando,
                'bubble_tech':bubble , 'strong_tech':strong,'weak_tech':weak
                }
        
        item['ph_len'] = len(item['ph_token'])
        return item

    def input_to_batch(self, item):
        item_names = [item['item_name']]
        text = [item['text']]
        ph = [item['ph']]
        txt_tokens = torch.LongTensor(item['ph_token'])[None, :].to(self.device)
        txt_lengths = torch.LongTensor([txt_tokens.shape[1]])[None, :].to(self.device)
        
        note = torch.LongTensor(item['note'])[None, :].to(self.device)
        note_dur = torch.FloatTensor(item['note_dur'])[None, :].to(self.device)
        note_type = torch.LongTensor(item['note_type'][:hparams['max_input_tokens']])[None, :].to(self.device)

        mix = torch.LongTensor(item['mix_tech'][:hparams['max_input_tokens']])[None, :].to(self.device)
        falsetto= torch.LongTensor(item['falsetto_tech'][:hparams['max_input_tokens']])[None, :].to(self.device)
        breathy = torch.LongTensor(item['breathy_tech'][:hparams['max_input_tokens']])[None, :].to(self.device)
        pharyngeal = torch.LongTensor(item['pharyngeal_tech'][:hparams['max_input_tokens']])[None, :].to(self.device)
        vibrato = torch.LongTensor(item['vibrato_tech'][:hparams['max_input_tokens']])[None, :].to(self.device)
        glissando = torch.LongTensor(item['glissando_tech'][:hparams['max_input_tokens']])[None, :].to(self.device)
        bubble = torch.LongTensor(item['bubble_tech'][:hparams['max_input_tokens']])[None, :].to(self.device)
        strong = torch.LongTensor(item['strong_tech'][:hparams['max_input_tokens']])[None, :].to(self.device)
        weak = torch.LongTensor(item['weak_tech'][:hparams['max_input_tokens']])[None, :].to(self.device)
        
        spk_id= torch.LongTensor([item['spk_id']]).to(self.device)

        batch = {
            'item_name': item_names,
            'text': text,
            'ph': ph,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'spk_id': spk_id,
            'notes': note,
            'note_durs': note_dur,
            'note_types': note_type
        }

        batch['mix'],batch['falsetto'],batch['breathy']=mix,falsetto,breathy
        batch['pharyngeal'],batch['vibrato'],batch['glissando']=pharyngeal,vibrato,glissando
        batch['bubble'],batch['strong'],batch['weak']=bubble,strong,weak
        return batch

    @classmethod
    def example_run(cls):
        set_hparams()
        exp_name = hparams['exp_name'].split('/')[-1]
        tech2id ={
            'control': ['0'],
            'mix': ['1'],
            'falsetto': ['2'],
            'breathy': ['3'],
            'vibrato': ['5'],
            'glissando': ['6'],
            'bubble': ['7'],
            'strong': ['8'],
            'weak': ['9']
        }
        infer_ins = cls(hparams)
        items_list = json.load(open(f"{hparams['processed_data_dir']}/metadata.json"))
        spker_set = json.load(open(f"{hparams['processed_data_dir']}/spker_set.json", 'r'))
        item_name = "Chinese#ZH-Alto-1#Mixed_Voice_and_Falsetto#十年#Mixed_Voice_Group#0000"
        # item_name = 'English#EN-Alto-1#Breathy#all is found#Breathy_Group#0000'
        inp = {'gen': item_name}
        
        for item in items_list:
            if inp['gen'] in item['item_name']:
                inp['text_gen']=item['ph']
                inp['note_gen']=item['ep_pitches']
                inp['note_dur_gen'] =item['ep_notedurs']
                inp['note_type_gen']=item['ep_types']  
                singer = item['singer']
                inp['spk_id']=spker_set[singer]
                break
        infer_dir = f'infer_out/{exp_name}/{item_name}'
        os.makedirs(infer_dir, exist_ok=True)
        for tech, ref_tech in tech2id.items():
            wav_fn = os.path.join(infer_dir, f'{tech}.wav')
            inp['tech_list'] = ref_tech * len(inp['text_gen'])
            out = infer_ins.infer_once(inp)
            wav_out, mel_out = out
            save_wav(wav_out, wav_fn, hparams['audio_sample_rate'])
            print(f'enjoy {wav_fn}')

if __name__ == '__main__':
    techinfer.example_run()