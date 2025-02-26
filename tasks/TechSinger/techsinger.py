from modules.TechSinger.techsinger import RFSinger, RFPostnet
from tasks.TechSinger.base_gen_task import AuxDecoderMIDITask, f0_to_figure
from utils.commons.hparams import hparams
import torch
from utils.commons.ckpt_utils import load_ckpt
from tasks.TechSinger.dataset import TechDataset
import torch.nn.functional as F
from utils.commons.tensor_utils import tensors_to_scalars
from utils.audio.pitch.utils import denorm_f0

# stage 1
class RFSingerTask(AuxDecoderMIDITask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = TechDataset
        self.mse_loss_fn = torch.nn.MSELoss()
        self.drop_prob=hparams['drop_tech_prob']

    def build_tts_model(self):
        dict_size = len(self.token_encoder)
        self.model = RFSinger(dict_size, hparams)
        self.gen_params = [p for p in self.model.parameters() if p.requires_grad]

    def drop_multi(self, tech, drop_p):
        if torch.rand(1) < drop_p:
            tech = torch.ones_like(tech, dtype=tech.dtype) * 2
        elif torch.rand(1) < drop_p:
            random_tech = torch.rand_like(tech, dtype=torch.float32)
            tech[random_tech < drop_p] = 2
        return tech
            
    def run_model(self, sample, infer=False):
        txt_tokens = sample["txt_tokens"]
        mel2ph = sample["mel2ph"]
        spk_id = sample["spk_ids"]
        f0, uv = sample["f0"], sample["uv"]
        notes, note_durs, note_types = sample["notes"], sample["note_durs"], sample["note_types"]
        
        target = sample["mels"]
        if not infer:
            tech_drop = {
                'mix': self.drop_prob,
                'falsetto': self.drop_prob,
                'breathy': self.drop_prob,
                'bubble': self.drop_prob,
                'strong': self.drop_prob,
                'weak': self.drop_prob,
                'glissando': self.drop_prob,
                'pharyngeal': self.drop_prob,
                'vibrato': self.drop_prob
            }
            for tech, drop_p in tech_drop.items():
                sample[tech] = self.drop_multi(sample[tech], drop_p)
        
        mix, falsetto, breathy=sample['mix'], sample['falsetto'], sample['breathy']
        bubble,strong,weak=sample['bubble'],sample['strong'],sample['weak']
        pharyngeal, vibrato, glissando = sample['pharyngeal'], sample['vibrato'], sample['glissando']
        output = self.model(txt_tokens, mel2ph=mel2ph, spk_id=spk_id, f0=f0, uv=uv, 
                            note=notes, note_dur=note_durs, note_type=note_types,
                            mix=mix, falsetto=falsetto, breathy=breathy,
                            bubble=bubble, strong=strong, weak=weak,
                            pharyngeal=pharyngeal, vibrato=vibrato, glissando=glissando, 
                            infer=infer)
        
        losses = {}
        
        self.add_mel_loss(output['mel_out'], target, losses)
        self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
        self.add_pitch_loss(output, sample, losses)
        
        return losses, output

    def add_pitch_loss(self, output, sample, losses):
        mel2ph = sample['mel2ph']  # [B, T_s]
        f0 = sample['f0']
        uv = sample['uv']
        nonpadding = (mel2ph != 0).float()
        if hparams["f0_gen"] == "flow":
            losses["pflow"] = output["pflow"]
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                output["uv_pred"][:, :, 0], uv, reduction='none') * nonpadding).sum() / nonpadding.sum() * hparams['lambda_uv']
        else:
            losses["fdiff"] = output["fdiff"]
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                output["uv_pred"][:, :, 0], uv, reduction='none') * nonpadding).sum() / nonpadding.sum() * hparams['lambda_uv']
            
    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {"flow": 0}
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            outputs['losses'], model_out = self.run_model(sample, infer=True)
            outputs['total_loss'] = sum(outputs['losses'].values())
            sr = hparams["audio_sample_rate"]
            gt_f0 = denorm_f0(sample['f0'], sample["uv"])
            wav_gt = self.vocoder.spec2wav(sample["mels"][0].cpu().numpy(), f0=gt_f0[0].cpu().numpy())
            self.logger.add_audio(f'wav_gt_{batch_idx}', wav_gt, self.global_step, sr)

            wav_pred = self.vocoder.spec2wav(model_out['mel_out'][0].cpu().numpy(), f0=model_out["f0_denorm_pred"][0].cpu().numpy())
            self.logger.add_audio(f'wav_pred_{batch_idx}', wav_pred, self.global_step, sr)
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'][0], f'mel_{batch_idx}')
            self.logger.add_figure(
                f'f0_{batch_idx}',
                f0_to_figure(gt_f0[0], None, model_out["f0_denorm_pred"][0]),
                self.global_step)
        return outputs

def self_clone(x):
    if x == None:
        return None
    y = x.clone()
    result = torch.cat((x, y), dim=0)
    return result

# stage 2
class RFPostnetTask(RFSingerTask):
    def __init__(self):
        super(RFPostnetTask, self).__init__()
        
    def build_model(self):
        self.build_pretrain_model()
        self.model = RFPostnet()

    def build_pretrain_model(self):
        dict_size = len(self.token_encoder)
        self.pretrain = RFSinger(dict_size, hparams)
        from utils.commons.ckpt_utils import load_ckpt
        load_ckpt(self.pretrain, hparams['fs2_ckpt_dir'], 'model', strict=True) 
        for k, v in self.pretrain.named_parameters():
            v.requires_grad = False    
    
    def run_model(self, sample, infer=False, noise=None):
        txt_tokens = sample["txt_tokens"]
        mel2ph = sample["mel2ph"]
        spk_id = sample["spk_ids"]
        f0, uv = sample["f0"], sample["uv"]
        notes, note_durs, note_types = sample["notes"], sample["note_durs"], sample["note_types"]
        target = sample["mels"]
        cfg = False
        
        if infer:
            cfg = True
            mix, falsetto, breathy = sample['mix'],sample['falsetto'],sample['breathy']
            bubble,strong,weak=sample['bubble'],sample['strong'],sample['weak']
            pharyngeal,vibrato,glissando = sample['pharyngeal'],sample['vibrato'],sample['glissando']
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
            
            output = self.pretrain(txt_tokens, mel2ph=mel2ph, spk_id=spk_id, f0=f0, uv=uv, 
                                note=notes, note_dur=note_durs, note_type=note_types,
                                mix=mix, falsetto=falsetto, breathy=breathy,
                                bubble=bubble, strong=strong, weak=weak,
                                pharyngeal=pharyngeal, vibrato=vibrato, glissando=glissando, 
                                infer=infer)
        else:
            tech_drop = {
                'mix': self.drop_prob,
                'falsetto': self.drop_prob,
                'breathy': self.drop_prob,
                'bubble': self.drop_prob,
                'strong': self.drop_prob,
                'weak': self.drop_prob,
                'glissando': self.drop_prob,
                'pharyngeal': self.drop_prob,
                'vibrato': self.drop_prob,
            }
            for tech, drop_p in tech_drop.items():
                sample[tech] = self.drop_multi(sample[tech], drop_p)
            mix, falsetto, breathy = sample['mix'],sample['falsetto'],sample['breathy']
            bubble,strong,weak=sample['bubble'],sample['strong'],sample['weak']
            pharyngeal,vibrato,glissando = sample['pharyngeal'],sample['vibrato'],sample['glissando']
            output = self.pretrain(txt_tokens, mel2ph=mel2ph, spk_id=spk_id, f0=f0, uv=uv, 
                                note=notes, note_dur=note_durs, note_type=note_types,
                                mix=mix, falsetto=falsetto, breathy=breathy,
                                bubble=bubble, strong=strong, weak=weak,
                                pharyngeal=pharyngeal, vibrato=vibrato, glissando=glissando, 
                                infer=infer)

        self.model(target, infer, output, cfg, cfg_scale=hparams['cfg_scale'],  noise=noise)
        losses = {}
        losses["flow"] = output["flow"]
        return losses, output

    def build_optimizer(self, model):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams['lr'],
            betas=(0.9, 0.98),
            eps=1e-9)
        return self.optimizer

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=0.5)
