import numpy as np
import torch
from torch import nn, optim  # , distributed
from torch.optim import lr_scheduler
from torch.backends import cudnn
from torch.utils import data
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from . import model, dataset, ranger, benchmark


# run: python -m torch.distributed.launch main.py


def main():
    # seed = 1283
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    cudnn.benchmark = True
    cudnn.deterministic = True
    writer = SummaryWriter(log_dir="/home/mist/output", flush_secs=30)

    # distributed
    # distributed.init_process_group(backend="nccl")
    # local_rank = torch.distributed.get_rank()
    # torch.cuda.set_device(local_rank)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_folder = dataset.RadarDataset(train=True)
    valid_folder = dataset.RadarDataset(train=False)

    # distributed
    # train = data.DataLoader(train_folder, batch_size=3, shuffle=False,
    #                         pin_memory=True, num_workers=2, sampler=DistributedSampler(train_folder))
    # valid = data.DataLoader(valid_folder, batch_size=1, shuffle=False,
    #                         pin_memory=True, num_workers=2, sampler=DistributedSampler(valid_folder))

    train = data.DataLoader(train_folder, batch_size=1, shuffle=False, pin_memory=True, num_workers=20)
    valid = data.DataLoader(valid_folder, batch_size=1, shuffle=False, pin_memory=True, num_workers=5)

    generator = model.ConvLSTMNetwork(10).cuda()
    generator_loss_func = model.weightedLoss().cuda()
    generator_optimizer = ranger.Ranger(generator.parameters())
    generator, generator_optimizer = \
        benchmark.load(generator, generator_optimizer, "./checkpoint_e20.pth.tar")

    # generator.eval()
    # writer.add_graph(generator, torch.rand([1, 10, 1, 512, 512]).cuda())
    # w

    # distributed
    # generator = torch.nn.parallel.DistributedDataParallel(
    # generator, device_ids=[local_rank], output_device=local_rank)

    discriminator = model.Discriminator().cuda()
    discriminator_loss_func = nn.BCEWithLogitsLoss().cuda()
    discriminator_optimizer = optim.SGD(discriminator.parameters(), lr=1e-2)
    discriminator, discriminator_optimizer = \
        benchmark.load(discriminator, discriminator_optimizer, "./checkpoint_discriminator.pth.tar")
    discriminator.train()

    # discriminator.eval()
    # writer.add_graph(discriminator, torch.rand([1, 10, 1, 512, 512]).cuda())

    generator_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(generator_optimizer, 10)
    generator.train()
    # discriminator.train()
    # sample, _ = train_folder.correcter()
    # writer.add_images('inputs_hmr', sample[:, 1, ...].unsqueeze(1), 0)
    # writer.add_images('inputs_radar', sample[:, 0, ...].unsqueeze(1), 0)
    print(generator)
    print(discriminator)
    for epoch in range(51):
        t = tqdm(train, leave=False, total=len(train))
        train_loss = []
        train_acc = []
        verify_loss = []
        for i, (targetVar, inputVar) in enumerate(t):
            # print(inputVar.size(), targetVar.size())
            inputs = inputVar.cuda()  # B,S,C,H,W

            generator_optimizer.zero_grad()
            generator_pred = generator(inputs)  # B,S,C,H,W -> fake

            discriminator_optimizer.zero_grad()
            discriminator_pred_fake = discriminator(generator_pred)
            discriminator_loss = discriminator_loss_func(discriminator_pred_fake, torch.zeros([1]).cuda())
            discriminator_loss_aver = discriminator_loss.item()
            discriminator_loss.backward(retain_graph=True)

            label = targetVar.cuda()  # B,S,C,H,W
            generator_loss = generator_loss_func(generator_pred, label)  # + discriminator_loss
            generator_loss_aver = generator_loss.item()
            generator_loss.backward()
            generator_optimizer.step()
            _, p, f, c, _ = benchmark.acc(label, generator_pred)

            t.set_postfix({
                'tL': '{:.6f}'.format(generator_loss_aver),
                'dL': '{:.6f}'.format(discriminator_loss_aver),
                'ep': '{:02d}'.format(epoch),
                'last': '{:.2f}'.format(torch.mean(label.reshape(10, -1).sum(1))),
            })

            train_loss.append(generator_loss_aver)
            verify_loss.append(discriminator_loss_aver)

            discriminator_pred_truth = discriminator(label)
            discriminator_loss_2 = discriminator_loss_func(discriminator_pred_truth, torch.ones([1]).cuda())

            if i % 5 == 0:
                discriminator_loss_2.backward()
                discriminator_optimizer.step()

            total_l = (discriminator_loss + discriminator_loss_2) / 2
            writer.add_scalar('Loss/Discriminator', total_l, epoch * len(train) + i + 1)
            writer.add_scalar('Loss/Train', generator_loss_aver, epoch * len(train) + i + 1)

            writer.add_scalar('POD/Train', p, epoch * len(train) + i + 1)
            writer.add_scalar('FAR/Train', f, epoch * len(train) + i + 1)
            writer.add_scalar('ETS/Train', c, epoch * len(train) + i + 1)

            # tl = aa + bb + cc + dd
            # writer.add_scalar('Factor/A, TP', aa / tl, epoch * len(train) + i + 1)
            # writer.add_scalar('Factor/B, FP', bb / tl, epoch * len(train) + i + 1)
            # writer.add_scalar('Factor/C, FN', cc / tl, epoch * len(train) + i + 1)
            # writer.add_scalar('Factor/D, TN', dd / tl, epoch * len(train) + i + 1)
            torch.cuda.empty_cache()

        with torch.no_grad():
            generator.eval()
            valid_loss = []
            t = tqdm(valid, leave=False, total=len(valid))
            for i, (targetVar, inputVar) in enumerate(t):
                inputs = inputVar.cuda()
                label = targetVar.cuda()
                pred = generator(inputs)

                loss = generator_loss_func(pred, label)
                _, p, f, c, _ = benchmark.acc(label, pred)
                loss_aver = loss.item()
                t.set_postfix({
                    'vL': '{:.6f}'.format(loss_aver),
                    'ep': '{:02d}'.format(1),
                    'last': '{:.2f}'.format(torch.mean(label.reshape(10, -1).sum(1))),
                })
                valid_loss.append(loss_aver)
                if i == 5:
                    # draw images
                    # pred[pred < 0.01] = 0
                    # writer.add_images('inputs_hmr', inputs[0, :, 1, ...], epoch)
                    # writer.add_images('inputs', inputs[0, :, 1, ...], epoch)
                    writer.add_images('inputs', inputs[0, ...], epoch)
                    writer.add_images('labels', label[0, ...], epoch)
                    writer.add_images('outputs', pred[0, ...], epoch)

                writer.add_scalar('Loss/Valid', np.mean(loss_aver), epoch * len(valid) + i + 1)
                writer.add_scalar('POD/Valid', p, epoch * len(valid) + i + 1)
                writer.add_scalar('FAR/Valid', f, epoch * len(valid) + i + 1)
                writer.add_scalar('ETS/Valid', c, epoch * len(valid) + i + 1)

        generator_scheduler.step(epoch)
        print("epoch: {}, loss: {:.6f}".
              format(epoch, np.mean(valid_loss)))
        if epoch % 4 == 0 and epoch != 0:
            print("Saving checkpoint...")
            state = {
                "epoch": epoch,
                "generator_model": generator.state_dict(),
                "generator_optimizer": generator_optimizer.state_dict(),
                "discriminator_model": discriminator.state_dict(),
                "discriminator_optimizer": discriminator_optimizer.state_dict()
            }
            torch.save(state, "./checkpoint_e{}.pth.tar".format(epoch))
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
