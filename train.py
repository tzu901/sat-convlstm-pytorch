from src.dataset import DigitDataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, cuda, optim
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
from tqdm import tqdm
import numpy as np
import argparse
from model import ED
from encoder import Encoder
from decoder import Decoder
import matplotlib.pyplot as plt

INPUT_FRAMES = 3
OUTPUT_FRAMES = 3
EPOCH = 10000
BATCH_SIZE = 4
DEVICE = 'cuda' if cuda.is_available() else 'cpu'
LEARNING_RATE = 0.0001

TIMESTAMP = "2020-03-09T00-00-00"
parser = argparse.ArgumentParser()
parser.add_argument('-clstm',
                    '--convlstm',
                    help='use convlstm as base cell',
                    action='store_true')
parser.add_argument('-cgru',
                    '--convgru',
                    help='use convgru as base cell',
                    action='store_true')
parser.add_argument('--batch_size',
                    default=4,
                    type=int,
                    help='mini-batch size')
parser.add_argument('-lr', default=0.0001, type=float, help='G learning rate')
parser.add_argument('-frames_input',
                    default=3,
                    type=int,
                    help='sum of input frames')
parser.add_argument('-frames_output',
                    default=3,
                    type=int,
                    help='sum of predict frames')
parser.add_argument('-epochs', default=10000, type=int, help='sum of epochs')
args = parser.parse_args()

save_dir = './save_model/' + TIMESTAMP

if args.convlstm:
    encoder_params = convlstm_encoder_params
    decoder_params = convlstm_decoder_params
if args.convgru:
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params
else:
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params

def train(train_dataloader, validation_dataloader):
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
    net = ED(encoder, decoder).cuda()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):

        print('==> loading existing model')
        model_info = torch.load(os.path.join(save_dir, 'checkpoin.pth.tar'))
        net.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(net.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cur_epoch = 0

    lossfunction = nn.MSELoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    for epoch in range(cur_epoch, args.epochs + 1):
        t = tqdm(train_dataloader, leave=False, total=len(train_dataloader))
        for i, (idx, targetVar, inputVar) in enumerate(t):
            inputs = inputVar.to(device)
            label = targetVar.to(device) 
        
            optimizer.zero_grad()
            net.train()
            
            pred = net(inputs)
            
            loss = lossfunction(pred, label)

            loss_aver = loss.item() / args.batch_size
            train_losses.append(loss_aver)
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
            optimizer.step()
            t.set_postfix({
                'trainloss': '{:.6f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
            })
        
        with torch.no_grad():
            net.eval()
            t = tqdm(validation_dataloader, leave=False, total=len(validation_dataloader))
            for i, (idx, targetVar, inputVar) in enumerate(t):
                if i == 3000:
                    break
                inputs = inputVar.to(device)
                label = targetVar.to(device)
                pred = net(inputs)
                loss = lossfunction(pred, label)
                loss_aver = loss.item() / args.batch_size
                # record validation loss
                valid_losses.append(loss_aver)
                # print ("validloss: {:.6f},  epoch : {:02d}".format(loss_aver,epoch),end = '\r', flush=True)
                t.set_postfix({
                    'validloss': '{:.6f}'.format(loss_aver),
                    'epoch': '{:02d}'.format(epoch)
                })

        torch.cuda.empty_cache()

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        epoch_len = len(str(args.epochs))

        train_losses = []
        valid_losses = []

        if (epoch + 1) % 10 == 0:
            # 繪製並儲存損失圖
            plt.figure(figsize=(10,5))
            plt.title("Training and Validation Loss")
            plt.plot(avg_train_losses, label="train")
            plt.plot(avg_valid_losses, label="valid")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(f"{save_dir}/loss_plot_epoch_{epoch}.png")
            plt.close()
    
            # 儲存模型
            state = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth.tar'))

    

    with open("avg_train_losses.txt", 'wt') as f:
        for i in avg_train_losses:
            print(i, file=f)

    with open("avg_valid_losses.txt", 'wt') as f:
        for i in avg_valid_losses:
            print(i, file=f)

if __name__ == '__main__':

    training_dataset = DigitDataset(
        r'./txt/train.txt',
        INPUT_FRAMES,
        OUTPUT_FRAMES,
        transform=transforms.ToTensor()
    )


    validation_dataset = DigitDataset(
        r'./txt/valid.txt',
        INPUT_FRAMES,
        OUTPUT_FRAMES,
        transform=transforms.ToTensor()
    )

    train_dataloader = DataLoader(
        training_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    train(train_dataloader, validation_dataloader)

    
