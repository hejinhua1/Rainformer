import torch
from SwinTransformer import StageModule, StageModule_up, StageModule_up_final
from torch import nn, optim
from tool import *
from tqdm import tqdm, trange

class Net(nn.Module):
    def __init__(self, input_channel, hidden_dim, downscaling_factors, layers, heads, head_dim, window_size, relative_pos_embedding):
        super(Net, self).__init__()
        self.stage1 = StageModule(in_channels=input_channel, hidden_dimension=hidden_dim, layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding, h_w=[72, 72])

        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding, h_w=[36, 36])

        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding, h_w=[18, 18])

        self.stage4 = StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding, h_w=[9, 9])

        # ----------------------------------------------------------------------------------------

        self.stage5 = StageModule_up(in_channels=hidden_dim * 8, hidden_dimension=hidden_dim * 4,
                                     layers=layers[3], upscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                     window_size=window_size, relative_pos_embedding=relative_pos_embedding, h_w=[18, 18])

        self.stage6 = StageModule_up(in_channels=hidden_dim * 8, hidden_dimension=hidden_dim * 2,
                                     layers=layers[2], upscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                     window_size=window_size, relative_pos_embedding=relative_pos_embedding, h_w=[36, 36])

        self.stage7 = StageModule_up(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 1,
                                     layers=layers[1], upscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                     window_size=window_size, relative_pos_embedding=relative_pos_embedding, h_w=[72, 72])

        self.stage8 = StageModule_up_final(in_channels=hidden_dim * 2, hidden_dimension=input_channel,
                                     layers=layers[0], upscaling_factor=downscaling_factors[0], num_heads=heads[0],
                                     head_dim=head_dim,
                                     window_size=window_size, relative_pos_embedding=relative_pos_embedding, h_w=[288, 288])

    def forward(self, x):
        x1 = self.stage1(x)     # (4, 96, 72, 72)
        x2 = self.stage2(x1)    # (4, 192, 36, 36)
        x3 = self.stage3(x2)    # (4, 384, 18, 18)
        x4 = self.stage4(x3)    # (4, 768, 9, 9)

        x5 = self.stage5(x4, x3)    # (4, 768, 18, 18)
        x6 = self.stage6(x5, x2)    # (4, 384, 36, 36)
        x7 = self.stage7(x6, x1)    # (4, 192, 72, 72)
        x8 = self.stage8(x7)        # (4, 9, 288, 288)

        return x8

if __name__ == '__main__':
    file_idx = 1
    epoch_size, batch_size = 200, 24
    x = torch.randn((batch_size, 9, 288, 288))
    y = torch.randn((batch_size, 9, 288, 288))
    net = Net(
        input_channel=9,
        hidden_dim=96,
        downscaling_factors=(4, 2, 2, 2),
        layers=(2, 2, 2, 2),
        heads=(3, 6, 12, 24),
        head_dim=32,
        window_size=9,
        relative_pos_embedding=True,)
    device = torch.device('cpu')
    train_seq, test_seq = np.load('../train_seq.npy'), np.load('../test_seq.npy')
    val_seq = train_seq[5000:]
    train_seq = train_seq[:5000]
    opt = optim.Adam(net.parameters(), lr=1e-3)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=5, verbose=True)
    criterion = BMAEloss().to(device)


    for epoch in range(1, epoch_size + 1):
        train_l_sum, test_l_sum, n = 0.0, 0.0, 0
        net.train()
        np.random.shuffle(train_seq)
        ran = np.arange(batch_size, train_seq.shape[0], batch_size)
        pbar = tqdm(ran)
        for batch in pbar:
            x = x.to(device)
            y = y.to(device)
            y_hat = net(x)
            loss = criterion(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_num = loss.detach().cpu().numpy()
            pbar.set_description('Train MSE Loss: ' + str(loss_num / batch_size))
            train_l_sum += loss_num
            n += batch_size
        train_loss = train_l_sum / n







