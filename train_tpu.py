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

# === XLA / TPU ===
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from meldataset import build_dataloader
from models import *
from losses import *
from utils import *
from optimizers import build_optimizer

class MyDataParallel(torch.nn.DataParallel):
    # Không dùng trên TPU; giữ lại để tương thích khi gọi thuộc tính
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

# --------- core train function (per TPU process) ----------
def _mp_fn(index, config_path):
    # Mỗi process có seed khác nhau để tránh trùng mẫu
    torch.manual_seed(1234 + index)
    np.random.seed(1234 + index)
    random.seed(1234 + index)

    # XLA device
    device = xm.xla_device()

    config = yaml.safe_load(open(config_path, "r", encoding="utf-8"))

    log_dir = config['log_dir']
    is_master = xm.is_master_ordinal()
    if is_master:
        os.makedirs(log_dir, exist_ok=True)
        shutil.copy(config_path, os.path.join(log_dir, os.path.basename(config_path)))

    # SummaryWriter & logger chỉ ở master
    writer = None
    if is_master:
        writer = SummaryWriter(log_dir + "/tensorboard")
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
        symbol_dict = {s: i for i, s in enumerate(symbols)}
        n_token = len(symbol_dict) + 1
        if is_master:
            print("\nFound:", n_token, "symbols")
    except Exception as e:
        print(f"\nERROR: Cannot find {e} in config file!\nYour config is likely outdated.")
        raise SystemExit(1)

    loss_params = Munch(config['loss_params'])
    optimizer_params = Munch(config['optimizer_params'])

    train_list, val_list = get_data_path_list(train_path, val_path)

    if is_master:
        print("\nInitializing train_dataloader")

    # Lưu ý: build_dataloader trả về DataLoader thường
    # Trên TPU, bọc bằng MpDeviceLoader để stream sang XLA device.
    # (Nếu muốn sharding theo core, cập nhật build_dataloader để gắn DistributedSampler)
    train_dataloader = build_dataloader(
        train_list, root_path, symbol_dict,
        batch_size=batch_size,
        num_workers=0,  # TPU khuyến nghị num_workers nhỏ
        dataset_config={"debug": debug},
        device="cpu"  # dữ liệu sẽ được push sang XLA qua MpDeviceLoader
    )

    if is_master:
        print("Initializing val_dataloader")
    val_dataloader = build_dataloader(
        val_list, root_path, symbol_dict,
        batch_size=batch_size,
        validation=True,
        num_workers=0,
        dataset_config={"debug": debug},
        device="cpu"
    )

    # Bọc dataloader cho XLA
    train_loader = pl.MpDeviceLoader(train_dataloader, device)
    val_loader = pl.MpDeviceLoader(val_dataloader, device)

    # build model
    model_params = recursive_munch(config['model_params'])
    model_params['n_token'] = n_token
    model = build_model(model_params)
    _ = [model[key].to(device) for key in model]

    # KHÔNG dùng DataParallel trên TPU
    # (PJRT/XLA mỗi core là một process riêng)
    # Giữ nguyên tham chiếu để code phía dưới không bị vỡ
    # nhưng không quấn lại bằng MyDataParallel.

    start_epoch = 0
    iters = 0

    load_pretrained = config.get('pretrained_model', '') != ''

    gl = GeneratorLoss(model.mpd, model.msd).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd).to(device)

    # schedulers per module
    scheduler_params = {
        "max_lr": optimizer_params.lr,
        "pct_start": float(0),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }
    scheduler_params_dict = {key: scheduler_params.copy() for key in model}
    scheduler_params_dict['decoder']['max_lr'] = optimizer_params.ft_lr * 2
    scheduler_params_dict['style_encoder']['max_lr'] = optimizer_params.ft_lr * 2

    optimizer = build_optimizer(
        {key: model[key].parameters() for key in model},
        scheduler_params_dict=scheduler_params_dict,
        lr=optimizer_params.lr
    )

    # adjust acoustic module learning rate
    for module in ["decoder", "style_encoder"]:
        for g in optimizer.optimizers[module].param_groups:
            g['betas'] = (0.0, 0.99)
            g['lr'] = optimizer_params.ft_lr
            g['initial_lr'] = optimizer_params.ft_lr
            g['min_lr'] = 0
            g['weight_decay'] = 1e-4

    # load pretrained
    if load_pretrained:
        try:
            training_strats = config['training_strats']
        except Exception:
            if is_master:
                print("\nNo training_strats found in config. Proceeding with default settings...")
            training_strats = {'ignore_modules': '', 'freeze_modules': ''}
        model, optimizer, start_epoch, iters = load_checkpoint(
            model, optimizer, config['pretrained_model'],
            load_only_params=config.get('load_only_params', True),
            ignore_modules=training_strats['ignore_modules'],
            freeze_modules=training_strats['freeze_modules']
        )
    else:
        raise Exception('Must have a pretrained!')

    n_down = model.text_aligner.n_down

    best_loss = float('inf')
    iters = int(iters) if iters is not None else 0

    stft_loss = MultiResolutionSTFTLoss().to(device)

    if is_master:
        print('\ndecoder', optimizer.optimizers['decoder'])

    ############################################## TRAIN ##############################################
    for epoch in range(start_epoch, epochs):
        running_loss = 0.0
        start_time = time.time()

        _ = [model[key].eval() for key in model]
        model.text_aligner.train()
        model.text_encoder.train()
        model.predictor.train()
        model.msd.train()
        model.mpd.train()

        for i, batch in enumerate(train_loader):
            waves = batch[0]
            # chuyển lên XLA device
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
            except Exception:
                # bắt buộc mark_step để XLA commit dù skip
                xm.mark_step()
                continue

            mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
            s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

            # encode
            t_en = model.text_encoder(texts, input_lengths, text_mask)

            # 50% chance monotonic
            if bool(random.getrandbits(1)):
                asr = (t_en @ s2s_attn)
            else:
                asr = (t_en @ s2s_attn_mono)

            d_gt = s2s_attn_mono.sum(axis=-1).detach()

            # style
            s = model.style_encoder(mels.unsqueeze(1))

            d, p = model.predictor(
                t_en, s, input_lengths, s2s_attn_mono, text_mask
            )

            mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
            en, gt, p_en, wav = [], [], [], []

            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)
                random_start = np.random.randint(0, mel_length - mel_len)
                en.append(asr[bib, :, random_start:random_start + mel_len])
                p_en.append(p[bib, :, random_start:random_start + mel_len])
                gt.append(mels[bib, :, (random_start * 2):((random_start + mel_len) * 2)])
                y = waves[bib][(random_start * 2) * 300:((random_start + mel_len) * 2) * 300]
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

            # reshape F0_real (giữ nguyên như code gốc)
            F0_real = F0_real.view(batch_size, -1)

            loss_F0_rec = (F.smooth_l1_loss(F0_real, F0_fake)) / 10
            loss_norm_rec = F.smooth_l1_loss(N_real, N_fake)

            # --- Discriminator update ---
            optimizer.zero_grad()
            d_loss = dl(wav.detach(), y_rec.detach()).mean()
            d_loss.backward()
            # với XLA: nên mark_step sau khi gọi optimizer.step()
            optimizer.step('msd')
            optimizer.step('mpd')
            xm.mark_step()

            # --- Generator update ---
            optimizer.zero_grad()
            loss_mel = stft_loss(y_rec, wav)
            loss_gen_all = gl(wav, y_rec).mean()

            loss_ce = 0
            loss_dur = 0
            for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                _s2s_pred = _s2s_pred[:_text_length, :]
                _text_input = _text_input[:_text_length].long()
                _s2s_trg = torch.zeros_like(_s2s_pred)
                for pidx in range(_s2s_trg.shape[0]):
                    _s2s_trg[pidx, :_text_input[pidx]] = 1
                _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)

                loss_dur += F.l1_loss(_dur_pred[1:_text_length - 1], _text_input[1:_text_length - 1])
                loss_ce += F.binary_cross_entropy_with_logits(_s2s_pred.flatten(), _s2s_trg.flatten())

            loss_ce /= texts.size(0)
            loss_dur /= texts.size(0)

            loss_s2s = 0
            for _s2s_pred, _text_input, _text_length in zip(s2s_pred, texts, input_lengths):
                loss_s2s += F.cross_entropy(_s2s_pred[:_text_length], _text_input[:_text_length])
            loss_s2s /= texts.size(0)

            loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10

            g_loss = (
                loss_params.lambda_mel * loss_mel +
                loss_params.lambda_F0 * loss_F0_rec +
                loss_params.lambda_ce * loss_ce +
                loss_params.lambda_norm * loss_norm_rec +
                loss_params.lambda_dur * loss_dur +
                loss_params.lambda_gen * loss_gen_all +
                loss_params.lambda_mono * loss_mono +
                loss_params.lambda_s2s * loss_s2s
            )

            running_loss += float(loss_mel.item())
            g_loss.backward()
            if torch.isnan(g_loss):
                raise RuntimeError("NaN loss detected")

            optimizer.step('predictor')
            optimizer.step('style_encoder')
            optimizer.step('decoder')
            optimizer.step('text_encoder')
            optimizer.step('text_aligner')
            xm.mark_step()

            iters += 1

            if is_master and ((i + 1) % log_interval == 0):
                logger.info(
                    'Epoch [%d/%d], Step [%d/%d], Mel Loss: %.5f, Disc Loss: %.5f, Dur Loss: %.5f, CE Loss: %.5f, Norm Loss: %.5f, F0 Loss: %.5f, Gen Loss: %.5f, S2S Loss: %.5f, Mono Loss: %.5f'
                    % (epoch + 1, epochs, i + 1, len(train_list) // batch_size,
                       running_loss / log_interval, d_loss, loss_dur, loss_ce,
                       loss_norm_rec, loss_F0_rec, loss_gen_all, loss_s2s, loss_mono)
                )
                writer.add_scalar('train/mel_loss', running_loss / log_interval, iters)
                writer.add_scalar('train/gen_loss', loss_gen_all, iters)
                writer.add_scalar('train/d_loss', d_loss, iters)
                writer.add_scalar('train/ce_loss', loss_ce, iters)
                writer.add_scalar('train/dur_loss', loss_dur, iters)
                writer.add_scalar('train/norm_loss', loss_norm_rec, iters)
                writer.add_scalar('train/F0_loss', loss_F0_rec, iters)
                running_loss = 0.0
                if is_master:
                    print('Time elapsed:', time.time() - start_time)

            # checkpoint tạm
            if (iters % 1000 == 0) and is_master:
                state = {
                    'net': {key: model[key].state_dict() for key in model},
                    'optimizer': optimizer.state_dict(),
                    'iters': iters,
                    'val_loss': 0,
                    'epoch': epoch,
                }
                save_path = os.path.join(log_dir, 'current_model.pth')
                # dùng xm.save để an toàn trên TPU
                xm.save(state, save_path)

        ############################################## EVAL ##############################################
        if is_master:
            print("\nEvaluating...")
        loss_test = 0.0
        loss_align = 0.0
        loss_f = 0.0
        _ = [model[key].eval() for key in model]

        with torch.no_grad():
            iters_test = 0
            for batch_idx, batch in enumerate(val_loader):
                try:
                    optimizer.zero_grad()
                    waves = batch[0]
                    batch = [b.to(device) for b in batch[1:]]
                    texts, input_lengths, mels, mel_input_length = batch

                    mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                    text_mask = length_to_mask(input_lengths).to(texts.device)

                    _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                    s2s_attn = s2s_attn.transpose(-1, -2)
                    s2s_attn = s2s_attn[..., 1:]
                    s2s_attn = s2s_attn.transpose(-1, -2)

                    mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                    s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                    t_en = model.text_encoder(texts, input_lengths, text_mask)
                    asr = (t_en @ s2s_attn_mono)

                    d_gt = s2s_attn_mono.sum(axis=-1).detach()

                    s = model.style_encoder(mels.unsqueeze(1))
                    d, p = model.predictor(t_en, s, input_lengths, s2s_attn_mono, text_mask)

                    mel_len = int(mel_input_length.min().item() / 2 - 1)
                    en, gt, p_en, wav = [], [], [], []

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item() / 2)
                        random_start = np.random.randint(0, mel_length - mel_len)
                        en.append(asr[bib, :, random_start:random_start + mel_len])
                        p_en.append(p[bib, :, random_start:random_start + mel_len])
                        gt.append(mels[bib, :, (random_start * 2):((random_start + mel_len) * 2)])
                        y = waves[bib][(random_start * 2) * 300:((random_start + mel_len) * 2) * 300]
                        wav.append(torch.from_numpy(y).to(device))

                    wav = torch.stack(wav).float().detach()
                    en = torch.stack(en)
                    p_en = torch.stack(p_en)
                    gt = torch.stack(gt).detach()

                    s = model.style_encoder(gt.unsqueeze(1))
                    F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s)

                    loss_dur = 0.0
                    for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                        _s2s_pred = _s2s_pred[:_text_length, :]
                        _text_input = _text_input[:_text_length].long()
                        _s2s_trg = torch.zeros_like(_s2s_pred)
                        for bib in range(_s2s_trg.shape[0]):
                            _s2s_trg[bib, :_text_input[bib]] = 1
                        _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)
                        loss_dur += F.l1_loss(_dur_pred[1:_text_length - 1], _text_input[1:_text_length - 1])

                    loss_dur /= texts.size(0)

                    y_rec = model.decoder(en, F0_fake, N_fake, s)
                    loss_mel = stft_loss(y_rec.squeeze(), wav.detach())

                    F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
                    loss_F0 = F.l1_loss(F0_real, F0_fake) / 10

                    loss_test += float(loss_mel)
                    loss_align += float(loss_dur)
                    loss_f += float(loss_F0)

                    iters_test += 1
                except Exception:
                    xm.mark_step()
                    continue

        # Reduce/average metric giữa các cores
        loss_test = xm.mesh_reduce('loss_test', loss_test, lambda x: sum(x) / len(x))
        loss_align = xm.mesh_reduce('loss_align', loss_align, lambda x: sum(x) / len(x))
        loss_f = xm.mesh_reduce('loss_f', loss_f, lambda x: sum(x) / len(x))

        if is_master:
            print('Epochs:', epoch + 1)
            logger.info('Validation loss: %.3f, Dur loss: %.3f, F0 loss: %.3f\n\n\n' %
                        (loss_test / max(iters_test, 1), loss_align / max(iters_test, 1), loss_f / max(iters_test, 1)))
            writer.add_scalar('eval/mel_loss', loss_test / max(iters_test, 1), epoch + 1)
            writer.add_scalar('eval/dur_loss', loss_align / max(iters_test, 1), epoch + 1)
            writer.add_scalar('eval/F0_loss', loss_f / max(iters_test, 1), epoch + 1)

            # Save best/periodic
            if (epoch + 1) % save_freq == 0:
                if (loss_test / max(iters_test, 1)) < best_loss:
                    best_loss = loss_test / max(iters_test, 1)
                print('Saving..')
                state = {
                    'net': {key: model[key].state_dict() for key in model},
                    'optimizer': optimizer.state_dict(),
                    'iters': iters,
                    'val_loss': loss_test / max(iters_test, 1),
                    'epoch': epoch,
                }
                save_path = os.path.join(log_dir, f'epoch_{epoch:05d}.pth')
                xm.save(state, save_path)

    if is_master:
        print("Training finished.")

@click.command()
@click.option('-p', '--config_path', default='Configs/config.yaml', type=str)
def main(config_path):
    # Launch 1 process per TPU core (thường là 8)
    xmp.spawn(_mp_fn, args=(config_path,), nprocs=None, start_method='fork')

if __name__=="__main__":
    main()
