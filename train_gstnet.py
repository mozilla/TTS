import argparse
import os
import sys
import time
import traceback
import datetime

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from datasets.TTSDataset import MyDataset
from distribute import (DistributedSampler, apply_gradient_allreduce,
                        init_distributed, reduce_tensor)
from layers.losses import L1LossMasked, MSELossMasked
from utils.audio import AudioProcessor
from utils.generic_utils import (NoamLR, check_update, count_parameters,
                                 create_experiment_folder, get_git_branch,
                                 load_config, remove_experiment_folder,
                                 weight_decay,
                                 set_init_dict, copy_config_file, setup_model,
                                 split_dataset, gradual_training_scheduler)
from utils.logger import Logger
from utils.speakers import load_speaker_mapping, save_speaker_mapping, \
    get_speakers
from utils.synthesis import synthesis
from utils.text.symbols import phonemes, symbols
from utils.visual import plot_alignment, plot_spectrogram
from datasets.preprocess import get_preprocessor_by_name

from models.gstnet import GSTNet
from ranger.ranger import Ranger

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(54321)
use_cuda = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
print(" > Using CUDA: ", use_cuda)
print(" > Number of GPUs: ", num_gpus)

def setup_loader(ap, is_val=False, verbose=False):
    global meta_data_train
    global meta_data_eval
    if "meta_data_train" not in globals():
        if c.meta_file_train is not None:
            meta_data_train = get_preprocessor_by_name(c.dataset)(c.data_path, c.meta_file_train)
        else:
            meta_data_train = get_preprocessor_by_name(c.dataset)(c.data_path)
    if "meta_data_eval" not in globals() and c.run_eval:
        if c.meta_file_val is not None:
            meta_data_eval = get_preprocessor_by_name(c.dataset)(c.data_path, c.meta_file_val)
        else:
            meta_data_eval, meta_data_train = split_dataset(meta_data_train)
    if is_val and not c.run_eval:
        loader = None
    else:
        dataset = MyDataset(
            c.r,
            c.text_cleaner,
            meta_data=meta_data_eval if is_val else meta_data_train,
            ap=ap,
            batch_group_size=0 if is_val else c.batch_group_size * c.batch_size,
            min_seq_len=c.min_seq_len,
            max_seq_len=c.max_seq_len,
            phoneme_cache_path=c.phoneme_cache_path,
            use_phonemes=c.use_phonemes,
            phoneme_language=c.phoneme_language,
            enable_eos_bos=c.enable_eos_bos_chars,
            verbose=verbose)
        sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        loader = DataLoader(
            dataset,
            batch_size=c.eval_batch_size if is_val else c.batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            drop_last=False,
            sampler=sampler,
            num_workers=c.num_val_loader_workers
            if is_val else c.num_loader_workers,
            pin_memory=False)
    return loader


def save_checkpoint(model, optimizer, model_loss, out_path,
                    current_step, epoch):
    checkpoint_path = 'checkpoint_{}.pth.tar'.format(current_step)
    checkpoint_path = os.path.join(out_path, checkpoint_path)
    print(" | | > Checkpoint saving : {}".format(checkpoint_path))

    new_state_dict = model.state_dict()
    state = {
        'model': new_state_dict,
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'step': current_step,
        'epoch': epoch,
        'linear_loss': model_loss,
        'date': datetime.date.today().strftime("%B %d, %Y")
    }
    torch.save(state, checkpoint_path)


def save_best_model(model, optimizer, model_loss, best_loss, out_path,
                    current_step, epoch):
    if model_loss < best_loss:
        new_state_dict = model.state_dict()
        state = {
            'model': new_state_dict,
            'optimizer': optimizer.state_dict(),
            'step': current_step,
            'epoch': epoch,
            'linear_loss': model_loss,
            'date': datetime.date.today().strftime("%B %d, %Y")
        }
        best_loss = model_loss
        bestmodel_path = 'best_model.pth.tar'
        bestmodel_path = os.path.join(out_path, bestmodel_path)
        print("\n > BEST MODEL ({0:.5f}) : {1:}".format(
            model_loss, bestmodel_path))
        torch.save(state, bestmodel_path)
    return best_loss

def setup_gstnet_model(model_tacogst, c):
    model = GSTNet(model_tacogst.gst.encoder.recurrence.input_size,
                   model_tacogst.gst.encoder.recurrence.hidden_size,
                   model_tacogst.gst.style_token_layer.attention.W_value.out_features)

    return model

def train(model, model_tacogst, criterion, optimizer, scheduler,
          ap, global_step, epoch):
    data_loader = setup_loader(ap, is_val=False, verbose=(epoch == 0))
    if c.use_speaker_embedding:
        speaker_mapping = load_speaker_mapping(OUT_PATH)
    model.train()
    epoch_time = 0
    avg_loss = 0
    avg_step_time = 0
    avg_loader_time = 0
    print("\n > Epoch {}/{}".format(epoch, c.epochs), flush=True)
    if use_cuda:
        batch_n_iter = int(len(data_loader.dataset) / (c.batch_size * num_gpus))
    else:
        batch_n_iter = int(len(data_loader.dataset) / c.batch_size)
    end_time = time.time()
    for num_iter, data in enumerate(data_loader):
        start_time = time.time()

        # setup input data
        text_input = data[0]
        text_lengths = data[1]
        speaker_names = data[2]
        linear_input = data[3]
        mel_input = data[4]
        mel_lengths = data[5]
        stop_targets = data[6]
        avg_text_length = torch.mean(text_lengths.float())
        avg_spec_length = torch.mean(mel_lengths.float())
        loader_time = time.time() - end_time

        if c.use_speaker_embedding:
            speaker_ids = [speaker_mapping[speaker_name]
                           for speaker_name in speaker_names]
            speaker_ids = torch.LongTtrain(ensor(speaker_ids)
        else:
            speaker_ids = None

        global_step += 1

        # setup lr
        if c.lr_decay:
            scheduler.step()
        optimizer.zero_grad()

        # dispatch data to GPU
        if use_cuda:
            text_input = text_input.cuda(non_blocking=True)
            text_lengths = text_lengths.cuda(non_blocking=True)
            mel_input = mel_input.cuda(non_blocking=True)
            mel_lengths = mel_lengths.cuda(non_blocking=True)
            linear_input = linear_input.cuda(non_blocking=True)
            stop_targets = stop_targets.cuda(non_blocking=True)
            if speaker_ids is not None:
                speaker_ids = speaker_ids.cuda(non_blocking=True)

        gst_style_ground_truth, gst_enc_ground_truth = model_tacogst.gst(mel_input)
        inputs = model_tacogst.embedding(text_input)
        encoder_output = model_tacogst.encoder(inputs)

        # forward pass model
        gst_style = model(encoder_output, speaker_ids=speaker_ids)

        loss = criterion(gst_style, gst_style_ground_truth.squeeze().data)
        loss.backward()

        optimizer, current_lr = weight_decay(optimizer, c.wd)
        grad_norm, _ = check_update(model, c.grad_clip)
        optimizer.step()

        step_time = time.time() - start_time
        epoch_time += step_time

        if global_step % c.print_step == 0:
            print(
                "   | > Step:{}/{}  GlobalStep:{}  TotalLoss:{:.5f} GradNorm:{:.5f} "
                "  AvgTextLen:{:.1f}  AvgSpecLen:{:.1f}  StepTime:{:.2f}  "
                "LoaderTime:{:.2f}  LR:{:.6f}".format(
                    num_iter, batch_n_iter, global_step, loss.item(),
                    grad_norm, avg_text_length, avg_spec_length, step_time,
                    loader_time, current_lr),
                flush=True)

        # aggregate losses from processes
        if num_gpus > 1:
            loss = reduce_tensor(loss.data, num_gpus)

        if args.rank == 0:
            avg_step_time += step_time
            avg_loader_time += loader_time
            avg_loss += loss

            # Plot Training Iter Stats
            # reduce TB load
            if global_step % 10 == 0:
                iter_stats = {"loss": loss.item(),
                              "lr": current_lr,
                              "grad_norm": grad_norm,
                              "step_time": step_time}
                tb_logger.tb_train_iter_stats(global_step, iter_stats)

            if global_step % c.save_step == 0:
                if c.checkpoint:
                    # save model
                    save_checkpoint(model, optimizer,
                                    loss.item(), OUT_PATH, global_step,
                                    epoch)

                # Diagnostic visualizations
                # const_spec = postnet_output[0].data.cpu().numpy()
                # gt_spec = linear_input[0].data.cpu().numpy() if c.model in ["Tacotron", "TacotronGST"] else  mel_input[0].data.cpu().numpy()
                # align_img = alignments[0].data.cpu().numpy()
                #
                # figures = {
                #     "prediction": plot_spectrogram(const_spec, ap),
                #     "ground_truth": plot_spectrogram(gt_spec, ap),
                #     "alignment": plot_alignment(align_img)
                # }
                # tb_logger.tb_train_figures(global_step, figures)
                #
                # # Sample audio
                # if c.model in ["Tacotron", "TacotronGST"]:
                #     train_audio = ap.inv_spectrogram(const_spec.T)
                # else:
                #     train_audio = ap.inv_mel_spectrogram(const_spec.T)
                # tb_logger.tb_train_audios(global_step,
                #                           {'TrainAudio': train_audio},
                #                           c.audio["sample_rate"])
        end_time = time.time()

    avg_loss /= (num_iter + 1)
    avg_step_time /= (num_iter + 1)
    avg_loader_time /= (num_iter + 1)

    # print epoch stats
    print(
        "   | > EPOCH END -- GlobalStep:{}  AvgTotalLoss:{:.5f}  "
        " EpochTime:{:.2f}  "
        "AvgStepTime:{:.2f}  AvgLoaderTime:{:.2f}".format(global_step, avg_loss,
                                                          epoch_time, avg_step_time,
                                                          avg_loader_time),
        flush=True)

    # Plot Epoch Stats
    if args.rank == 0:
        # Plot Training Epoch Stats
        epoch_stats = {"loss": avg_loss,
                       "epoch_time": epoch_time}
        tb_logger.tb_train_epoch_stats(global_step, epoch_stats)
        if c.tb_model_param_stats:
            tb_logger.tb_model_weights(model, global_step)
    return avg_loss, global_step

def evaluate(model, model_tacogst, criterion, ap, global_step, epoch):
    data_loader = setup_loader(ap, is_val=True)
    if c.use_speaker_embedding:
        speaker_mapping = load_speaker_mapping(OUT_PATH)
    model.eval()
    epoch_time = 0
    avg_loss = 0
    print("\n > Validation")
    with torch.no_grad():
        if data_loader is not None:
            for num_iter, data in enumerate(data_loader):
                start_time = time.time()
                
                # setup input data
                text_input = data[0]
                text_lengths = data[1]
                speaker_names = data[2]
                linear_input = data[3]
                mel_input = data[4]
                mel_lengths = data[5]
                stop_targets = data[6]
                
                if c.use_speaker_embedding:
                    speaker_ids = [speaker_mapping[speaker_name]
                                   for speaker_name in speaker_names]
                    speaker_ids = torch.LongTensor(speaker_ids)
                else:
                    speaker_ids = None
                
                # dispatch data to GPU
                if use_cuda:
                    text_input = text_input.cuda()
                    mel_input = mel_input.cuda()
                    mel_lengths = mel_lengths.cuda()
                    linear_input = linear_input.cuda()
                    stop_targets = stop_targets.cuda()
                    if speaker_ids is not None:
                        speaker_ids = speaker_ids.cuda()

                gst_style_ground_truth, gst_enc_ground_truth = model_tacogst.gst(mel_input)
                inputs = model_tacogst.embedding(text_input)
                encoder_output = model_tacogst.encoder(inputs)
        
                # forward pass model
                gst_style = model(encoder_output, speaker_ids=speaker_ids)


                # loss computation
                loss = criterion(gst_style, gst_style_ground_truth.squeeze().detach())
                
                step_time = time.time() - start_time
                epoch_time += step_time
                
                if num_iter % c.print_step == 0:
                    print(
                        "   | > TotalLoss: {:.5f} ".format(loss.item()),
                        flush=True)
                        
                # aggregate losses from processes
                if num_gpus > 1:
                    loss = reduce_tensor(loss.data, num_gpus)

                if args.rank == 0:
                    avg_loss += loss
            
            if args.rank == 0:
                # compute average loss
                avg_loss /= (num_iter + 1)
                
                # Plot Validation Stats
                epoch_stats = {"loss": loss}
                tb_logger.tb_eval_stats(global_step, epoch_stats)
    return avg_loss


#FIXME: move args definition/parsing inside of main?
def main(args): #pylint: disable=redefined-outer-name
    # Audio processor
    ap = AudioProcessor(**c.audio)

    # DISTRUBUTED
    if num_gpus > 1:
        init_distributed(args.rank, num_gpus, args.group_id,
                         c.distributed["backend"], c.distributed["url"])
    num_chars = len(phonemes) if c.use_phonemes else len(symbols)

    if c.use_speaker_embedding:
        speakers = get_speakers(c.data_path, c.meta_file_train, c.dataset)
        if args.restore_path:
            prev_out_path = os.path.dirname(args.restore_path)
            speaker_mapping = load_speaker_mapping(prev_out_path)
            assert all([speaker in speaker_mapping
                        for speaker in speakers]), "As of now you, you cannot " \
                                                   "introduce new speakers to " \
                                                   "a previously trained model."
        else:
            speaker_mapping = {name: i
                               for i, name in enumerate(speakers)}
        save_speaker_mapping(OUT_PATH, speaker_mapping)
        num_speakers = len(speaker_mapping)
        print("Training with {} speakers: {}".format(num_speakers,
                                                     ", ".join(speakers)))
    else:
        num_speakers = 0

    model_tacogst = setup_model(num_chars, num_speakers, c)
    checkpoint = torch.load(args.gst_model_path)
    model_tacogst.load_state_dict(checkpoint['model'])
    #model_tacogst.eval()
    
    model = setup_gstnet_model(model_tacogst, c)

    #optimizer = optim.Adam(model.parameters(), lr=c.lr, weight_decay=0)
    optimizer = Ranger(model.parameters())
    criterion = nn.L1Loss()

    if args.restore_path:
        checkpoint = torch.load(args.restore_path)
        try:
            # TODO: fix optimizer init, model.cuda() needs to be called before
            # optimizer restore
            # optimizer.load_state_dict(checkpoint['optimizer'])
            if c.reinit_layers:
                raise RuntimeError
            model.load_state_dict(checkpoint['model'])
        except:
            print(" > Partial model initialization.")
            model_dict = model.state_dict()
            model_dict = set_init_dict(model_dict, checkpoint, c)
            model.load_state_dict(model_dict)
            del model_dict
        print(
            " > Model restored from step %d" % checkpoint['step'], flush=True)
        args.restore_step = checkpoint['step']
    else:
        args.restore_step = 0

    if use_cuda:
        model = model.cuda()
        model_tacogst = model_tacogst.cuda()
        criterion.cuda()

    # DISTRUBUTED
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)

    if c.lr_decay:
        scheduler = NoamLR(
            optimizer,
            warmup_steps=c.warmup_steps,
            last_epoch=args.restore_step - 1)
    else:
        scheduler = None

    num_params = count_parameters(model)
    print("\n > Model has {} parameters".format(num_params), flush=True)

    if 'best_loss' not in locals():
        best_loss = float('inf')

    global_step = args.restore_step
    for epoch in range(0, c.epochs):
        train_loss, global_step = train(model, model_tacogst, criterion,
                                        optimizer, scheduler,
                                        ap, global_step, epoch)
        #TODO: add this
        val_loss = evaluate(model, model_tacogst, criterion, ap, global_step, epoch)
        print(
            " | > Training Loss: {:.5f}   Validation Loss: {:.5f}".format(
                train_loss, val_loss),
            flush=True)
        target_loss = train_loss
        if c.run_eval:
            target_loss = val_loss
        best_loss = save_best_model(model, optimizer, target_loss, best_loss,
                                    OUT_PATH, global_step, epoch)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--restore_path',
        type=str,
        help='Path to model outputs (checkpoint, tensorboard etc.).',
        default=0)
    parser.add_argument(
        '--gst_model_path',
        type=str,
        help='Path to the trained TacotronGST (checkpoint*.tar file).',
        default=0)
    parser.add_argument(
        '--config_path',
        type=str,
        help='Path to config file for training.',
    )
    parser.add_argument(
        '--debug',
        type=bool,
        default=True,
        help='Do not verify commit integrity to run training.')
    parser.add_argument(
        '--data_path',
        type=str,
        default='',
        help='Path to data directory (not preprocessed). It overrides config.json.')
    parser.add_argument(
        '--output_path',
        type=str,
        help='path for training outputs.',
        default='')
    parser.add_argument(
        '--output_folder',
        type=str,
        default='',
        help='folder name for training outputs.'
    )

    # DISTRUBUTED
    parser.add_argument(
        '--rank',
        type=int,
        default=0,
        help='DISTRIBUTED: process rank for distributed training.')
    parser.add_argument(
        '--group_id',
        type=str,
        default="",
        help='DISTRIBUTED: process group id.')
    args = parser.parse_args()

    # setup output paths and read configs
    c = load_config(args.config_path)
    _ = os.path.dirname(os.path.realpath(__file__))
    if args.data_path != '':
        c.data_path = args.data_path

    if args.output_path == '':
        OUT_PATH = os.path.join(_, c.output_path)
    else:
        OUT_PATH = args.output_path

    if args.group_id == '' and args.output_folder == '':
        OUT_PATH = create_experiment_folder(OUT_PATH, c.run_name, args.debug)
    else:
        OUT_PATH = os.path.join(OUT_PATH, args.output_folder)

    AUDIO_PATH = os.path.join(OUT_PATH, 'test_audios')

    if args.rank == 0:
        os.makedirs(AUDIO_PATH, exist_ok=True)
        new_fields = {}
        if args.restore_path:
            new_fields["restore_path"] = args.restore_path
        new_fields["github_branch"] = get_git_branch()
        copy_config_file(args.config_path, os.path.join(OUT_PATH, 'config.json'), new_fields)
        os.chmod(AUDIO_PATH, 0o775)
        os.chmod(OUT_PATH, 0o775)

    if args.rank == 0:
        LOG_DIR = OUT_PATH
        tb_logger = Logger(LOG_DIR)

    try:
        main(args)
    except KeyboardInterrupt:
        remove_experiment_folder(OUT_PATH)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0) #pylint: disable=protected-access
    except Exception: #pylint: disable=broad-except
        remove_experiment_folder(OUT_PATH)
        traceback.print_exc()
        sys.exit(1)
