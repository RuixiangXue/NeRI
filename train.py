from __future__ import print_function
import argparse
import os
import random
import shutil
from datetime import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from model import  Generator 
from dataset.dataset_rangeimages import RangeImgaes_Kitti_Dataset
import yaml
from utils import *
from dataset.range2pc import range2pc,write_ri
from dataset.write_ply import write_ply_o3d_geo,write_ply_o3d_normal
from extension.tools import pc_error
from extension.quant import *
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default='config/kitti_00.yaml')
    # Model Architecture Parameters
    parser.add_argument('--temporal_embed', type=str, default='1.25_80', help='base value/embed length for position encoding')
    parser.add_argument('--rotation_embed', type=str, default='1.25_80', help='base value/embed length for position encoding')
    parser.add_argument('--translation_embed', type=str, default='1.25_80', help='base value/embed length for position encoding')
    parser.add_argument('--stem_dim_num', type=str, default='1024_1', help='hidden dimension and length')
    parser.add_argument('--fc_hw_dim', type=str, default='4_125_26', help='out size (h,w) for mlp') #!!!!!!!!!!!
    parser.add_argument('--expansion', type=float, default=8, help='channel expansion from fc to conv')
    parser.add_argument('--reduction', type=int, default=2)
    parser.add_argument('--strides', type=int, nargs='+', default=[2, 2, 2, 2], help='strides list')
    parser.add_argument('--num_blocks', type=int, default=1)
    parser.add_argument('--norm', default='none', type=str, help='norm layer for generator', choices=['none', 'bn', 'in'])
    parser.add_argument('--act', type=str, default='gelu', help='activation to use', choices=['relu', 'leaky', 'leaky01', 'relu6', 'gelu', 'swish', 'softplus', 'hardswish'])
    parser.add_argument('--lower-width', type=int, default=32, help='lowest channel width for output feature maps')
    parser.add_argument("--single_res", default=True,action='store_true', help='single resolution,  added to suffix!!!!')
    parser.add_argument("--conv_type", default='conv', type=str,  help='upscale methods, can add bilinear and deconvolution methods', choices=['conv', 'deconv', 'bilinear'])
    # General training setups
    parser.add_argument('-j', '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('-b', '--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--not_resume_epoch', action='store_true', help='resuming start_epoch from checkpoint')
    parser.add_argument('-e', '--epochs', type=int, default=600, help='number of epochs to train for')
    parser.add_argument('--cycles', type=int, default=1, help='epoch cycles for training')
    parser.add_argument('--warmup', type=float, default=0.2, help='warmup epoch ratio compared to the epochs, default=0.2,  added to suffix!!!!')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
    parser.add_argument('--lr_type', type=str, default='cosine', help='learning rate type, default=cosine')
    parser.add_argument('--lr_steps', default=[], type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10,  added to suffix!!!!')
    parser.add_argument('--beta', type=float, default=0.5, help='beta for adam. default=0.5,  added to suffix!!!!')
    parser.add_argument('--loss_type', type=str, default='L1', help='loss type, default=L1')
    parser.add_argument('--lw', type=float, default=1.0, help='loss weight,  added to suffix!!!!')
    parser.add_argument('--sigmoid', action='store_true', help='using sigmoid for output prediction')
    # evaluation parameters
    parser.add_argument('--eval_only', action='store_true', default=False, help='do evaluation only')
    parser.add_argument('--eval_freq', type=int, default=50, help='evaluation frequency,  added to suffix!!!!')
    parser.add_argument('--quant_bit', type=int, default=-1, help='bit length for model quantization')
    parser.add_argument('--quant_mode', type=str, default='uniform', help='uniform or pw-1 or pw-2')
    parser.add_argument('--quant_axis', type=int, default=0, help='quantization axis (-1 means per tensor)')
    parser.add_argument('--eval_fps', action='store_true', default=False, help='fwd multiple times to test the fps ')
    parser.add_argument('--dump', action='store_true', default=False, help='dump the prediction images')
    # pruning paramaters
    parser.add_argument('--prune_steps', type=float, nargs='+', default=[0.,], help='prune steps')
    parser.add_argument('--prune_ratio', type=float, default=1.0, help='pruning ratio')
    # distribute learning parameters
    parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:9888', type=str,help='url used to set up distributed training')
    parser.add_argument('-d', '--distributed', action='store_true', default=False, help='distributed training,  added to suffix!!!!')
    # logging, output directory, 
    parser.add_argument('--debug', action='store_true', help='defbug status, earlier for train/eval')  
    parser.add_argument('-p', '--print-freq', default=100, type=int,)
    parser.add_argument('--weight', default='None', type=str, help='pretrained weights for ininitialization')
    parser.add_argument('--overwrite', action='store_true', help='overwrite the output dir if already exists')
    parser.add_argument('--outf', default='try_100f', help='folder to output images and model checkpoints')
    parser.add_argument('--suffix', default='', help="suffix str for outf")
    parser.add_argument('--frame_num', default='100', help="number of input frames")
    args = parser.parse_args()
    args.warmup = int(args.warmup * args.epochs)
    print(args)
    cfg = yaml.safe_load(open(args.cfg))
    torch.set_printoptions(precision=4) 
    if args.debug:
        args.eval_freq = 1
        args.outf = 'output/debug'
    else:
        args.outf = os.path.join('output', args.outf)

    if args.prune_ratio < 1 and not args.eval_only: 
        prune_str = '_Prune{}_{}'.format(args.prune_ratio, ','.join([str(x) for x in args.prune_steps]))
    else:
        prune_str = ''
    extra_str = '_Strd{}_{}Res{}{}'.format( ','.join([str(x) for x in args.strides]),  'Sin' if args.single_res else f'_lw{args.lw}_multi',  
            '_dist' if args.distributed else '', f'_eval' if args.eval_only else '')
    norm_str = '' if args.norm == 'none' else args.norm

    exp_id = f'{cfg["DATA"]["DATASET"]}/temporal_embed{args.temporal_embed}_translation_embed{args.translation_embed}_rotation_embed{args.rotation_embed}_{args.stem_dim_num}_fc_{args.fc_hw_dim}__exp{args.expansion}_reduce{args.reduction}_low{args.lower_width}_blk{args.num_blocks}_cycle{args.cycles}' + \
            f'_e{args.epochs}_warm{args.warmup}_b{args.batchSize}_{args.conv_type}_lr{args.lr}_{args.lr_type}_qbit{args.quant_bit}_qmode{args.quant_mode}' + \
            f'_{args.loss_type}{norm_str}{extra_str}{prune_str}'
    
    exp_id += f'_act{args.act}_{args.suffix}'
    args.exp_id = exp_id

    args.outf = os.path.join(args.outf, exp_id)
    if args.overwrite and os.path.isdir(args.outf):
        print('Will overwrite the existing output dir!')
        shutil.rmtree(args.outf)

    if not os.path.isdir(args.outf):
        os.makedirs(args.outf)

    port = hash(args.exp_id) % 20000 + 10000
    args.init_method =  f'tcp://127.0.0.1:{port}'
    print(f'init_method: {args.init_method}', flush=True)

    torch.set_printoptions(precision=2) 
    args.ngpus_per_node = torch.cuda.device_count()
    if args.distributed and args.ngpus_per_node > 1:
        mp.spawn(train, nprocs=args.ngpus_per_node, args=(args,))
    else:
        train(None, args)

def train(local_rank, args):
    cudnn.benchmark = True
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    train_best_psnr, val_best_psnr= [torch.tensor(0) for _ in range(2)] #, train_best_msssim , val_best_msssim 
    is_train_best, is_val_best = False, False

    Temporal_PE = PositionalEncoding(args.temporal_embed)
    Translation_PE = PositionalEncoding3d(args.translation_embed)
    Rotation_PE = PositionalEncoding3d(args.rotation_embed)
    args.temporal_embed_length = Temporal_PE.embed_length
    args.translation_embed_length = Translation_PE.embed_length
    args.rotation_embed_length = Rotation_PE.embed_length
    #args.temporal_embed_length+args.translation_embed_length+args.rotation_embed_length
    model = Generator(embed_length=args.temporal_embed_length+args.translation_embed_length+args.rotation_embed_length, stem_dim_num=args.stem_dim_num, fc_hw_dim=args.fc_hw_dim, expansion=args.expansion, 
        num_blocks=args.num_blocks, norm=args.norm, act=args.act, bias = True, reduction=args.reduction, conv_type=args.conv_type,
        stride_list=args.strides,  sin_res=args.single_res,  lower_width=args.lower_width, sigmoid=args.sigmoid)

    ##### prune model params and flops #####
    prune_net = args.prune_ratio < 1
    # import pdb; pdb.set_trace; from IPython import embed; embed()
    if prune_net:
        param_list = []
        for k,v in model.named_parameters():
            if 'weight' in k:
                if 'stem' in k:
                    stem_ind = int(k.split('.')[1])
                    param_list.append(model.stem[stem_ind])
                elif 'layers' in k[:6] and 'conv' in k:
                    layer_ind = int(k.split('.')[1])
                    param_list.append(model.layers[layer_ind].conv.conv)
        param_to_prune = [(ele, 'weight') for ele in param_list]
        prune_base_ratio = args.prune_ratio ** (1. / len(args.prune_steps))
        args.prune_steps = [int(x * args.epochs) for x in args.prune_steps]
        prune_num = 0
        if args.eval_only:
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num,
            )

    ##### get model params and flops #####
    total_params = sum([p.data.nelement() for p in model.parameters()]) / 1e6
    if local_rank in [0, None]:
        params = sum([p.data.nelement() for p in model.parameters()]) / 1e6

        print(f'{args}\n {model}\n Model Params: {params}M')
        with open('{}/rank0.txt'.format(args.outf), 'a') as f:
            f.write(str(model) + '\n' + f'Params: {params}M\n')
        writer = SummaryWriter(os.path.join(args.outf, f'param_{total_params}M', 'tensorboard'))
    else:
        writer = None

    # distrite model to gpu or parallel
    print("Use GPU: {} for training".format(local_rank))
    if args.distributed and args.ngpus_per_node > 1:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=args.init_method,
            world_size=args.ngpus_per_node,
            rank=local_rank,
        )
        torch.cuda.set_device(local_rank)
        assert torch.distributed.is_initialized()        
        args.batchSize = int(args.batchSize / args.ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.to(local_rank), device_ids=[local_rank], \
                                                          output_device=local_rank, find_unused_parameters=False)
    elif args.ngpus_per_node > 1:
        model = torch.nn.DataParallel(model).cuda() #model.cuda() #
    else:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), betas=(args.beta, 0.999))

    # resume from args.weight
    checkpoint = None
    loc = 'cuda:{}'.format(local_rank if local_rank is not None else 0)
    if args.weight != 'None':
        print("=> loading checkpoint '{}'".format(args.weight))
        checkpoint_path = args.weight
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        orig_ckt = checkpoint['state_dict']
        new_ckt={k.replace('blocks.0.',''):v for k,v in orig_ckt.items()} 
        if 'module' in list(orig_ckt.keys())[0] and not hasattr(model, 'module'):
            new_ckt={k.replace('module.',''):v for k,v in new_ckt.items()}
            model.load_state_dict(new_ckt)
        elif 'module' not in list(orig_ckt.keys())[0] and hasattr(model, 'module'):
            model.module.load_state_dict(new_ckt)
        else:
            model.load_state_dict(new_ckt)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.weight, checkpoint['epoch']))        

    # resume from model_latest
    checkpoint_path = os.path.join(args.outf, 'model_latest.pth')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if prune_net:
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num,
            )
            
            sparisity_num = 0.
            for param in param_list:
                sparisity_num += (param.weight == 0).sum()
            print(f'Model sparsity at Epoch{args.start_epoch}: {sparisity_num / 1e6 / total_params}')
        model.load_state_dict(checkpoint['state_dict'])
        print("=> Auto resume loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
    else:
        print("=> No resume checkpoint found at '{}'".format(checkpoint_path))

    args.start_epoch = 0
    if checkpoint is not None:
        args.start_epoch = checkpoint['epoch'] 
        train_best_psnr = checkpoint['train_best_psnr'].to(torch.device(loc))
        val_best_psnr = checkpoint['val_best_psnr'].to(torch.device(loc))
        optimizer.load_state_dict(checkpoint['optimizer'])

    if args.not_resume_epoch:
        args.start_epoch = 0
    # setup dataloader
    cfg = yaml.safe_load(open(args.cfg))
    Dataset_train = RangeImgaes_Kitti_Dataset(cfg,"train")
    Dataset_test = RangeImgaes_Kitti_Dataset(cfg,"test")

    train_dataloader = torch.utils.data.DataLoader(Dataset_train, batch_size=args.batchSize, shuffle=False,
         num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True,worker_init_fn=worker_init_fn)
   
    val_dataloader = torch.utils.data.DataLoader(Dataset_test, batch_size=args.batchSize, shuffle=False,
         num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True,worker_init_fn=worker_init_fn)

    data_size = len(Dataset_train)

    if args.eval_only:
        print('Evaluation ...')
        time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        print_str = f'{time_str}\t Results for checkpoint: {args.weight}\n'
        if prune_net:
            for param in param_to_prune:
                prune.remove(param[0], param[1])
            sparisity_num = 0.
            for param in param_list:
                sparisity_num += (param.weight == 0).sum()
            print_str += f'Model sparsity at Epoch{args.start_epoch}: {sparisity_num / 1e6 / total_params}\n'

        # import pdb; pdb.set_trace; from IPython import embed; embed()
        val_psnr= evaluate(model, val_dataloader, PE, local_rank, args) #, val_msssim 
        print_str += f'PSNR on validate set for bit {args.quant_bit} with axis {args.quant_axis}: {round(val_psnr.item(),2)}'
        print(print_str)
        with open('{}/eval.txt'.format(args.outf), 'a') as f:
            f.write(print_str + '\n\n')        
        return

    # Training
    start = datetime.now()
    total_epochs = args.epochs * args.cycles
    for epoch in range(args.start_epoch, total_epochs):
        model.train()
        ##### prune the network if needed #####
        if prune_net and epoch in args.prune_steps:
            prune_num += 1 
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num,
            )
            
            sparisity_num = 0.
            for param in param_list:
                sparisity_num += (param.weight == 0).sum()
            print(f'Model sparsity at Epoch{epoch}: {sparisity_num / 1e6 / total_params}')
        
        epoch_start_time = datetime.now()
        psnr_list = []
        # iterate over dataloader
        for i, (data,  norm_idx,pose) in enumerate(train_dataloader):
            if i > 10 and args.debug:
                break
            temporal_embed_input = Temporal_PE(norm_idx)
            rotation_embed_input = Rotation_PE(pose.squeeze(0)[0,:])
            translation_embed_input = Translation_PE(pose.squeeze(0)[1,:])
            #temporal_embed_input, rotation_embed_input, translation_embed_input
            embed_input = torch.cat((temporal_embed_input,rotation_embed_input, translation_embed_input), 1)

            if local_rank is not None:
                data = data.cuda(local_rank, non_blocking=True)
                embed_input = embed_input.cuda(local_rank, non_blocking=True)
            else:
                data,  embed_input = data.cuda(non_blocking=True).reshape(1,1,cfg["DATA"]["V_Res"],cfg["DATA"]["H_Res"]),   embed_input.cuda(non_blocking=True) #data.shape: torch.Size([1, 2000, 64, 1])

            # forward and backward
            output_list = model(embed_input) 

            #print(output_list[0].shape)
            target_list = [F.adaptive_avg_pool2d(data, x.shape[-2:]) for x in output_list] 
            loss_list = [loss_fn(output, target, args) for output, target in zip(output_list, target_list)]
            loss_list = [loss_list[i] * (args.lw if i < len(loss_list) - 1 else 1) for i in range(len(loss_list))]
            loss_sum = sum(loss_list)
            lr = adjust_lr(optimizer, epoch % args.epochs, i, data_size, args)
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

            # compute psnr and msssim
            psnr_list.append(psnr_fn(output_list, target_list))
            if i % args.print_freq == 0 or i == len(train_dataloader) - 1:
                train_psnr = torch.cat(psnr_list, dim=0) #(batchsize, num_stage)
                train_psnr = torch.mean(train_psnr, dim=0) #(num_stage)
                time_now_string = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                print_str = '[{}] Rank:{}, Epoch[{}/{}], Step [{}/{}], lr:{:.2e} PSNR: {}'.format(
                    time_now_string, local_rank, epoch+1, args.epochs, i+1, len(train_dataloader), lr, 
                    RoundTensor(train_psnr, 2, False))
                print(print_str, flush=True)
                if local_rank in [0, None]:
                    with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                        f.write(print_str + '\n')

        # collect numbers from other gpus
        if args.distributed and args.ngpus_per_node > 1:
            train_psnr = all_reduce([train_psnr.to(local_rank)])

        # ADD train_PSNR TO TENSORBOARD
        if local_rank in [0, None]:
            h, w = output_list[-1].shape[-2:]
            is_train_best = train_psnr[-1] > train_best_psnr
            train_best_psnr = train_psnr[-1] if train_psnr[-1] > train_best_psnr else train_best_psnr
            writer.add_scalar(f'Train/PSNR_{h}X{w}', train_psnr[-1].item(), epoch+1)
            writer.add_scalar(f'Train/best_PSNR_{h}X{w}', train_best_psnr.item(), epoch+1)
            print_str = '\t{}p: current: {:.2f}\t best: {:.2f}\t '.format(h, train_psnr[-1].item(), train_best_psnr.item())
            print(print_str, flush=True)
            with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n')
            writer.add_scalar('Train/lr', lr, epoch+1)
            epoch_end_time = datetime.now()
            print("Time/epoch: \tCurrent:{:.2f} \tAverage:{:.2f}".format( (epoch_end_time - epoch_start_time).total_seconds(), \
                    (epoch_end_time - start).total_seconds() / (epoch + 1 - args.start_epoch) ))

        state_dict = model.state_dict()
        save_checkpoint = {
            'epoch': epoch+1,
            'state_dict': state_dict,
            'train_best_psnr': train_best_psnr,
            'val_best_psnr': val_best_psnr,
            'optimizer': optimizer.state_dict(),   
        }    
        # evaluation
        if (epoch + 1) % args.eval_freq == 0 or epoch > total_epochs - 5:
            val_start_time = datetime.now()
            val_psnr= (evaluate(model, val_dataloader, Temporal_PE,Rotation_PE,Translation_PE, local_rank, args)).cuda()
            val_end_time = datetime.now()
            if args.distributed and args.ngpus_per_node > 1:
                val_psnr = all_reduce([val_psnr.to(local_rank)])          
            if local_rank in [0, None]:
                # ADD val_PSNR TO TENSORBOARD
                h, w = output_list[-1].shape[-2:]
                print_str = f'Eval best_PSNR at epoch{epoch+1}:'
                is_val_best = val_psnr[-1] > val_best_psnr
                val_best_psnr = val_psnr[-1] if is_val_best else val_best_psnr
                writer.add_scalar(f'Val/PSNR_{h}X{w}_gap{args.test_gap}', val_psnr[-1], epoch+1)
                writer.add_scalar(f'Val/best_PSNR_{h}X{w}_gap{args.test_gap}', val_best_psnr, epoch+1)
                print_str += '\t{}p: current: {:.2f}\tbest: {:.2f} \t\t Time/epoch: {:.2f}'.format(h, val_psnr[-1].item(),
                     val_best_psnr.item(),  (val_end_time - val_start_time).total_seconds())
                print(print_str)
                with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                    f.write(print_str + '\n')
                if is_val_best:
                    torch.save(save_checkpoint, '{}/model_val_best.pth'.format(args.outf))

        if local_rank in [0, None]:
            # state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(save_checkpoint, '{}/model_latest.pth'.format(args.outf))
            if is_train_best:
                torch.save(save_checkpoint, '{}/model_train_best.pth'.format(args.outf))

    print("Training complete in: " + str(datetime.now() - start))


@torch.no_grad()
def evaluate(model, val_dataloader, Temporal_PE,Rotation_PE,Translation_PE, local_rank, args):
    cfg = yaml.safe_load(open(args.cfg))
    # Model Quantization
    if args.quant_bit != -1:
        cur_ckt = model.state_dict()
        from dahuffman import HuffmanCodec
        quant_weitht_list = []
        args.wei_quant_scheme = args.quant_mode
        args.wei_bits = args.quant_bit
        args.bias_corr = False
        args.scale_bits = 0
        args.approximate = False
        args.break_point = 'norm'
        all_quant_error, all_quant_num = 0, 0
        all_tail_num = 0
        all_valid = 0 
        for each_layer,each_layer_weights in cur_ckt.items():
            if each_layer=='module.head_layers.3.bias':
                break
            #print('quantize for: %s, size: %s' % (each_layer, each_layer_weights.size()))
            #print('weights range: (%.4f, %.4f)' % (torch.min(each_layer_weights), torch.max(each_layer_weights)))
            w = each_layer_weights.clone().view(-1,1)
            w_int_tuple,qw, err, tail_num = quant_weights(w, args)
            each_layer_weights = qw.view(each_layer_weights.size())
            #print('dequantized weights range: (%.4f, %.4f)' % (torch.min(each_layer_weights), torch.max(each_layer_weights)))
            all_quant_error += err
            all_quant_num += len(qw)
            all_tail_num += tail_num
            cur_ckt[each_layer] = each_layer_weights
            valid = 0
            for w_int in w_int_tuple:
                valid += len(w_int)
                quant_weitht_list.append(w_int.flatten())
            all_valid += valid
        #print('layer quant RMSE: %.4e' % np.sqrt(all_quant_error / all_quant_num))
        #print('layer tail region percentage: %.2f' % (all_tail_num / all_quant_num * 100))     
        #print('len_valid_weight:',all_valid,"len_total",all_quant_num) 
        cat_param = torch.cat(quant_weitht_list)
        input_code_list = cat_param.tolist()
        unique, counts = np.unique(input_code_list, return_counts=True)
        num_freq = dict(zip(unique, counts))
        # generating HuffmanCoding table
        codec = HuffmanCodec.from_data(input_code_list)
        sym_bit_dict = {}
        for k, v in codec.get_code_table().items():
            sym_bit_dict[k] = v[0]
        total_bits = 0
        for num, freq in num_freq.items():
            total_bits += freq * sym_bit_dict[num]
        avg_bits = total_bits / len(input_code_list)    
        encoding_efficiency = avg_bits / args.quant_bit
        print_str = f'Entropy encoding efficiency for bit {args.quant_bit}: {encoding_efficiency}'
        print(print_str)
        if local_rank in [0, None]:
            with open('{}/eval.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n')       
        model.load_state_dict(cur_ckt)

    psnr_d1_list = []
    psnr_d2_list = []
    model.eval()
    for i, (data,  norm_idx,pose, pc, ori_pc, pc_seg) in enumerate(val_dataloader):
        if i > 10 and args.debug:
            break
        temporal_embed_input = Temporal_PE(norm_idx)
        rotation_embed_input = Rotation_PE(pose.squeeze(0)[0,:])
        translation_embed_input = Translation_PE(pose.squeeze(0)[1,:])
        #temporal_embed_input, rotation_embed_input, translation_embed_input
        embed_input = torch.cat((temporal_embed_input,rotation_embed_input, translation_embed_input), 1)
        if local_rank is not None:
            data = data.cuda(local_rank, non_blocking=True)
            embed_input = embed_input.cuda(local_rank, non_blocking=True)
        else:
            data,  embed_input = data.cuda(non_blocking=True), embed_input.cuda(non_blocking=True)

        start_time = datetime.now()
        output_list = model(embed_input)
        #print((datetime.now() - start_time).total_seconds())
        time_cost = (datetime.now() - start_time).total_seconds()
        torch.cuda.synchronize()
        if args.dump:
            visual_ri_dir = f'{args.outf}/visualize/rangeimage'
            print(f'Saving RIs to {visual_ri_dir}')
            if not os.path.isdir(visual_ri_dir):
                os.makedirs(visual_ri_dir)
            gt_ri_filename = os.path.join(visual_ri_dir, 'gt'+str(i) + '.png')
            pred_ri_filename = os.path.join(visual_ri_dir, 'pred'+str(i) + '.png')
            seg_ri_filename = os.path.join(visual_ri_dir, 'seg'+str(i) + '.png')
            write_ri(gt_ri_filename, data.squeeze(0))
            write_ri(pred_ri_filename, output_list[0].squeeze(0))
            write_ri(seg_ri_filename, pc_seg)
        
        output_pc_rec = range2pc(output_list[0].squeeze(0),args.cfg) #(64,2000,3)

        pc_seg = (pc_seg.permute(1,2,0)).numpy() 
        output_pc_rec = (output_pc_rec * pc_seg).reshape(-1,3) # (64*2000, 3)
        output_pc_rec = np.unique(output_pc_rec,axis=0)

        pc = pc.reshape(-1,3).numpy()
        pc = np.unique(pc,axis=0)
        
        visual_pc_dir = f'{args.outf}/visualize/pointcloud'
        if not os.path.isdir(visual_pc_dir):
            os.makedirs(visual_pc_dir)
        rec_pc_filename = os.path.join(visual_pc_dir, 'rec'+str(i) + '.ply')
        out_pc_filename = os.path.join(visual_pc_dir, 'out'+str(i) + '.ply')
        write_ply_o3d_normal(out_pc_filename, coords=pc, knn=20)
        write_ply_o3d_geo(rec_pc_filename, coords=output_pc_rec)

        pc_error_results = pc_error(out_pc_filename, rec_pc_filename,normal=True, resolution=None, show=False)
        if not args.dump:
            os.system('rm '+rec_pc_filename)
            os.system('rm '+out_pc_filename)
        else:
            print(f'Saving PCs to {visual_pc_dir}')
        
        psnr_d1 = torch.tensor(pc_error_results["mseF,PSNR (p2point)"]).unsqueeze(0)
        psnr_d2 = torch.tensor(pc_error_results['mseF,PSNR (p2plane)']).unsqueeze(0)

        psnr_d1_list.append(psnr_d1)
        psnr_d2_list.append(psnr_d2)

        fps = 1 / time_cost
        
        print_str = 'Rank:{},Step [{}/{}],PSNR_D1: {}, PSNR_D2: {}, FPS: {}'.format(
            local_rank, i, len(val_dataloader),
            RoundTensor(psnr_d1, 2, False), RoundTensor(psnr_d2, 2, False),
            round(fps, 2)
            ) 
        print(print_str)
        if local_rank in [0, None]:
            with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n')
            
    val_psnr_d1 = torch.cat(psnr_d1_list, dim=0)
    val_psnr_d1 = torch.mean(val_psnr_d1, dim=0)
    
    val_psnr_d2 = torch.cat(psnr_d2_list, dim=0)
    val_psnr_d2 = torch.mean(val_psnr_d2, dim=0)
    
    print_str = 'Frame_Num:{}, PSNR_D1_Avg: {}, PSNR_D2_Avg: {}'.format(
            len(val_dataloader),
            RoundTensor(val_psnr_d1, 2, False), RoundTensor(val_psnr_d2, 2, False)
            ) 
    print(print_str)
    if local_rank in [0, None]:
        with open('{}/rank0.txt'.format(args.outf), 'a') as f:
            f.write(print_str + '\n')
    
    model.train()

    return ((val_psnr_d1+val_psnr_d2)/2).unsqueeze(0)


if __name__ == '__main__':
    main()
