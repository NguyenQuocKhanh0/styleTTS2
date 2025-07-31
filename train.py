# load packages
import os
import random
import yaml
import time
from munch import Munch
import numpy as np
import torch
import torch.nn.functional as F
import click
import shutil
import warnings
warnings.simplefilter('ignore')
from torch.utils.tensorboard import SummaryWriter

from meldataset import build_dataloader

from models import *
from losses import *
from utils import *

from optimizers import build_optimizer

class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
import logging
from logging import StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


@click.command()
@click.option('-p', '--config_path', default='Configs/config.yaml', type=str)
def main(config_path):
    config = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    
    log_dir = config['log_dir']
    if not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(log_dir, os.path.basename(config_path)))
    writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)

    batch_size = config.get('batch_size', 10)
    debug = config.get('debug', True)
    epochs = config.get('epochs', 200)
    save_freq = config.get('save_freq', 2)
    log_interval = config.get('log_interval', 10)
    data_params = config.get('data_params', None)
    train_path = data_params['train_data']
    val_path = data_params['val_data']
    root_path = data_params['root_path']
    max_len = config.get('max_len', 200)

    try:
        symbols = (
                        list(config['symbol']['pad']) +
                        list(config['symbol']['punctuation']) +
                        list(config['symbol']['letters']) +
                        list(config['symbol']['letters_ipa']) +
                        list(config['symbol']['extend'])
                    )
        symbol_dict = {}
        for i in range(len((symbols))):
            symbol_dict[symbols[i]] = i

        n_token = len(symbol_dict) + 1
        print("\nFound:", n_token, "symbols")
    except Exception as e:
        print(f"\nERROR: Cannot find {e} in config file!\nYour config file is likely outdated, please download updated version from the repository.")
        raise SystemExit(1)
    
    loss_params = Munch(config['loss_params'])
    optimizer_params = Munch(config['optimizer_params'])
    
    train_list, val_list = get_data_path_list(train_path, val_path)
    device = 'cuda'

    print("\n")
    print("Initializing train_dataloader")
    train_dataloader = build_dataloader(train_list,
                                        root_path,
                                        symbol_dict,
                                        batch_size=batch_size,
                                        num_workers=3,
                                        dataset_config={"debug": debug},
                                        device=device)

    print("Initializing val_dataloader")
    val_dataloader = build_dataloader(val_list,
                                      root_path,
                                      symbol_dict,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=1,
                                      dataset_config={"debug": debug},
                                      device=device)
    
    # build model
    model_params = recursive_munch(config['model_params'])
    model_params['n_token'] = n_token
    model = build_model(model_params)
    _ = [model[key].to(device) for key in model]

    # DP
    for key in model:
        if key != "mpd" and key != "msd":
            model[key] = MyDataParallel(model[key])

    start_epoch = 0
    iters = 0

    load_pretrained = config.get('pretrained_model', '') != ''

    gl = GeneratorLoss(model.mpd, model.msd).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd).to(device)

    gl = MyDataParallel(gl)
    dl = MyDataParallel(dl)
    
    scheduler_params = {
        "max_lr": optimizer_params.lr,
        "pct_start": float(0),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }

    scheduler_params_dict= {key: scheduler_params.copy() for key in model}
    scheduler_params_dict['decoder']['max_lr'] = optimizer_params.ft_lr * 2
    scheduler_params_dict['style_encoder']['max_lr'] = optimizer_params.ft_lr * 2
    
    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                          scheduler_params_dict=scheduler_params_dict, lr=optimizer_params.lr)
        
    # adjust acoustic module learning rate
    for module in ["decoder", "style_encoder"]:
        for g in optimizer.optimizers[module].param_groups:
            g['betas'] = (0.0, 0.99)
            g['lr'] = optimizer_params.ft_lr
            g['initial_lr'] = optimizer_params.ft_lr
            g['min_lr'] = 0
            g['weight_decay'] = 1e-4
        
    # load models if there is a model
    if load_pretrained:
        try:
            training_strats = config['training_strats']
        except Exception as e:
            print("\nNo training_strats found in config. Proceeding with default settings...")
            training_strats = {}
            training_strats['ignore_modules'] = ''
            training_strats['freeze_modules'] = ''
        model, optimizer, start_epoch, iters = load_checkpoint(model,  optimizer, 
                                                               config['pretrained_model'], 
                                                               load_only_params=config.get('load_only_params', True),
                                                               ignore_modules=training_strats['ignore_modules'],
                                                               freeze_modules=training_strats['freeze_modules'])
    else:
        raise Exception('Must have a pretrained!')
        
    n_down = model.text_aligner.n_down

    best_loss = float('inf')  # best test loss
    iters = 0
    
    torch.cuda.empty_cache()
    
    stft_loss = MultiResolutionSTFTLoss().to(device)
    
    print('\ndecoder', optimizer.optimizers['decoder'])
    
############################################## TRAIN ##############################################

    for epoch in range(start_epoch, epochs):
        running_loss = 0
        start_time = time.time()

        _ = [model[key].eval() for key in model]
        
        model.text_aligner.train()
        model.text_encoder.train() 
        model.predictor.train()
        model.msd.train()
        model.mpd.train()

        for i, batch in enumerate(train_dataloader):
            waves = batch[0]
            batch = [b.to(device) for b in batch[1:]]
            texts, input_lengths, mels, mel_input_length = batch
            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                text_mask = length_to_mask(input_lengths).to(texts.device)
            try:
                ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)
                s2s_attn = s2s_attn.transpose(-1, -2)
                s2s_attn = s2s_attn[..., 1:]
                s2s_attn = s2s_attn.transpose(-1, -2)
            except:
                continue

            mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
            s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

            # encode
            t_en = model.text_encoder(texts, input_lengths, text_mask)
            
            # 50% of chance of using monotonic version
            if bool(random.getrandbits(1)):
                asr = (t_en @ s2s_attn)
            else:
                asr = (t_en @ s2s_attn_mono)

            d_gt = s2s_attn_mono.sum(axis=-1).detach()

            # compute the style of the entire utterance
            s = model.style_encoder(mels.unsqueeze(1))

            d, p = model.predictor(t_en, s, 
                                    input_lengths, 
                                    s2s_attn_mono, 
                                    text_mask)
                
            mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
            en = []
            gt = []
            p_en = []
            wav = []
            
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)

                random_start = np.random.randint(0, mel_length - mel_len)
                en.append(asr[bib, :, random_start:random_start+mel_len])
                p_en.append(p[bib, :, random_start:random_start+mel_len])
                gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])
                
                y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                wav.append(torch.from_numpy(y).to(device))
                
            wav = torch.stack(wav).float().detach()

            en = torch.stack(en)
            p_en = torch.stack(p_en)
            gt = torch.stack(gt).detach()
            
            s = model.style_encoder(gt.unsqueeze(1))           
                
            with torch.no_grad():
                F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
                N_real = log_norm(gt.unsqueeze(1)).squeeze(1)
                wav = wav.unsqueeze(1)

            F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s)

            y_rec = model.decoder(en, F0_fake, N_fake, s)

            loss_F0_rec =  (F.smooth_l1_loss(F0_real, F0_fake)) / 10
            loss_norm_rec = F.smooth_l1_loss(N_real, N_fake)

            optimizer.zero_grad()
            d_loss = dl(wav.detach(), y_rec.detach()).mean()
            d_loss.backward()
            optimizer.step('msd')
            optimizer.step('mpd')

            # generator loss
            optimizer.zero_grad()

            loss_mel = stft_loss(y_rec, wav)
            loss_gen_all = gl(wav, y_rec).mean()

            loss_ce = 0
            loss_dur = 0
            for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                _s2s_pred = _s2s_pred[:_text_length, :]
                _text_input = _text_input[:_text_length].long()
                _s2s_trg = torch.zeros_like(_s2s_pred)
                for p in range(_s2s_trg.shape[0]):
                    _s2s_trg[p, :_text_input[p]] = 1
                _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)

                loss_dur += F.l1_loss(_dur_pred[1:_text_length-1], 
                                       _text_input[1:_text_length-1])
                loss_ce += F.binary_cross_entropy_with_logits(_s2s_pred.flatten(), _s2s_trg.flatten())

            loss_ce /= texts.size(0)
            loss_dur /= texts.size(0)
            
            loss_s2s = 0
            for _s2s_pred, _text_input, _text_length in zip(s2s_pred, texts, input_lengths):
                loss_s2s += F.cross_entropy(_s2s_pred[:_text_length], _text_input[:_text_length])
            loss_s2s /= texts.size(0)

            loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10

            g_loss = loss_params.lambda_mel * loss_mel          +\
                     loss_params.lambda_F0 * loss_F0_rec        +\
                     loss_params.lambda_ce * loss_ce            +\
                     loss_params.lambda_norm * loss_norm_rec    +\
                     loss_params.lambda_dur * loss_dur          +\
                     loss_params.lambda_gen * loss_gen_all      +\
                     loss_params.lambda_mono * loss_mono        +\
                     loss_params.lambda_s2s * loss_s2s
            
            running_loss += loss_mel.item()
            g_loss.backward()
            if torch.isnan(g_loss):
                from IPython.core.debugger import set_trace
                set_trace()

            optimizer.step('predictor')
            optimizer.step('style_encoder')
            optimizer.step('decoder')
            
            optimizer.step('text_encoder')
            optimizer.step('text_aligner')

            iters = iters + 1
            
            if (i+1)%log_interval == 0:
                logger.info ('Epoch [%d/%d], Step [%d/%d], Mel Loss: %.5f, Disc Loss: %.5f, Dur Loss: %.5f, CE Loss: %.5f, Norm Loss: %.5f, F0 Loss: %.5f, Gen Loss: %.5f, S2S Loss: %.5f, Mono Loss: %.5f'
                    %(epoch+1, epochs, i+1, len(train_list)//batch_size, running_loss / log_interval, d_loss, loss_dur, loss_ce, loss_norm_rec, loss_F0_rec, loss_gen_all, loss_s2s, loss_mono))
                
                writer.add_scalar('train/mel_loss', running_loss / log_interval, iters)
                writer.add_scalar('train/gen_loss', loss_gen_all, iters)
                writer.add_scalar('train/d_loss', d_loss, iters)
                writer.add_scalar('train/ce_loss', loss_ce, iters)
                writer.add_scalar('train/dur_loss', loss_dur, iters)
                writer.add_scalar('train/norm_loss', loss_norm_rec, iters)
                writer.add_scalar('train/F0_loss', loss_F0_rec, iters)
                
                running_loss = 0
                
                print('Time elasped:', time.time()-start_time)

            if iters % 1000 == 0: # Save to current_model every 2000 iters
                state = {
                    'net':  {key: model[key].state_dict() for key in model}, 
                    'optimizer': optimizer.state_dict(),
                    'iters': iters,
                    'val_loss': 0,
                    'epoch': epoch,
                }
                save_path = os.path.join(log_dir, 'current_model.pth')
                torch.save(state, save_path)  


############################################## EVAL ##############################################


        print("\nEvaluating...")
        loss_test = 0
        loss_align = 0
        loss_f = 0
        _ = [model[key].eval() for key in model]

        with torch.no_grad():
            iters_test = 0
            for batch_idx, batch in enumerate(val_dataloader):
                optimizer.zero_grad()
                try:
                    waves = batch[0]
                    batch = [b.to(device) for b in batch[1:]]
                    texts, input_lengths, mels, mel_input_length = batch
                    with torch.no_grad():
                        mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                        text_mask = length_to_mask(input_lengths).to(texts.device)

                        _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                        s2s_attn = s2s_attn.transpose(-1, -2)
                        s2s_attn = s2s_attn[..., 1:]
                        s2s_attn = s2s_attn.transpose(-1, -2)

                        mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                        s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                        # encode
                        t_en = model.text_encoder(texts, input_lengths, text_mask)
                        asr = (t_en @ s2s_attn_mono)

                        d_gt = s2s_attn_mono.sum(axis=-1).detach()

                    # compute the style of the entire utterance
                    s = model.style_encoder(mels.unsqueeze(1))

                    d, p = model.predictor(t_en, s, 
                                            input_lengths, 
                                            s2s_attn_mono, 
                                            text_mask)
                    # get clips
                    mel_len = int(mel_input_length.min().item() / 2 - 1)
                    en = []
                    gt = []
                    p_en = []
                    wav = []

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item() / 2)

                        random_start = np.random.randint(0, mel_length - mel_len)
                        en.append(asr[bib, :, random_start:random_start+mel_len])
                        p_en.append(p[bib, :, random_start:random_start+mel_len])
                        gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])

                        y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                        wav.append(torch.from_numpy(y).to(device))

                    wav = torch.stack(wav).float().detach()

                    en = torch.stack(en)
                    p_en = torch.stack(p_en)
                    gt = torch.stack(gt).detach()

                    s = model.style_encoder(gt.unsqueeze(1)) 

                    F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s)

                    loss_dur = 0
                    for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                        _s2s_pred = _s2s_pred[:_text_length, :]
                        _text_input = _text_input[:_text_length].long()
                        _s2s_trg = torch.zeros_like(_s2s_pred)
                        for bib in range(_s2s_trg.shape[0]):
                            _s2s_trg[bib, :_text_input[bib]] = 1
                        _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)
                        loss_dur += F.l1_loss(_dur_pred[1:_text_length-1], 
                                                _text_input[1:_text_length-1])

                    loss_dur /= texts.size(0)

                    y_rec = model.decoder(en, F0_fake, N_fake, s)
                    loss_mel = stft_loss(y_rec.squeeze(), wav.detach())

                    F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1)) 

                    loss_F0 = F.l1_loss(F0_real, F0_fake) / 10

                    loss_test += (loss_mel).mean()
                    loss_align += (loss_dur).mean()
                    loss_f += (loss_F0).mean()

                    iters_test += 1
                except:
                    continue

        print('Epochs:', epoch + 1)
        logger.info('Validation loss: %.3f, Dur loss: %.3f, F0 loss: %.3f' % (loss_test / iters_test, loss_align / iters_test, loss_f / iters_test) + '\n\n\n')
        print('\n\n\n')
        writer.add_scalar('eval/mel_loss', loss_test / iters_test, epoch + 1)
        writer.add_scalar('eval/dur_loss', loss_test / iters_test, epoch + 1)
        writer.add_scalar('eval/F0_loss', loss_f / iters_test, epoch + 1)
        
        
        if (epoch + 1) % save_freq == 0 :
            if (loss_test / iters_test) < best_loss:
                best_loss = loss_test / iters_test
            print('Saving..')
            state = {
                'net':  {key: model[key].state_dict() for key in model}, 
                'optimizer': optimizer.state_dict(),
                'iters': iters,
                'val_loss': loss_test / iters_test,
                'epoch': epoch,
            }
            save_path = os.path.join(log_dir, 'epoch_%05d.pth' % epoch)
            torch.save(state, save_path)

                            
if __name__=="__main__":
    main()