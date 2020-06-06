import torch
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from . import model, ranger, dataset
from torch.utils import data


def acc(label, pred, threshold=1.0, times=10):
    # inputs: B, S, C, H, W
    bias_list = []
    pod_list = []
    far_list = []
    ets_list = []
    acc_list = []
    # print(label, pred)
    label = label.detach().squeeze() * 255
    pred = pred.detach().squeeze() * 255
    s1 = torch.tensor(threshold).cuda()
    s2 = torch.tensor(1.0).cuda()
    s3 = torch.tensor(0.0).cuda()
    default = torch.tensor(0.005).cuda()

    a = torch.sum(torch.where(pred[label >= s1] >= s1, s2, s3)) / times  # TP, A
    b = torch.sum(torch.where(pred[label < s1] >= s1, s2, s3)) / times  # FP, B
    c = torch.sum(torch.where(pred[label >= s1] < s1, s2, s3)) / times  # FN, C
    d = torch.sum(torch.where(pred[label < s1] < s1, s2, s3)) / times  # TN, D
    a = default if a == 0 else a
    b = default if b == 0 else b
    c = default if c == 0 else c
    d = default if d == 0 else d

    num = (a + b) * (a + c) / (a + c + b + d)
    ETS = (a - num) / (a + c + b - num)

    bias_list.append((a + b) / (a + c))
    pod_list.append(a / (a + b))
    far_list.append(c / (a + c))
    ets_list.append(ETS)
    acc_list.append((a + d) / (a + b + c + d))

    bias = torch.mean(torch.stack(bias_list)).cpu().numpy()
    pod = torch.mean(torch.stack(pod_list)).cpu().numpy()
    far = torch.mean(torch.stack(far_list)).cpu().numpy()
    ets = torch.mean(torch.stack(ets_list)).cpu().numpy()
    acc = torch.mean(torch.stack(acc_list)).cpu().numpy()
    return [bias, pod, far, ets, acc]


def load(net, optimizer, filename='checkpoin.pth.tar'):
    try:
        model_info = torch.load(filename)
        net.load_state_dict(model_info['generator_model'])
        if optimizer:
            optimizer.load_state_dict(model_info['generator_optimizer'])
    except Exception as e:
        print(e)
    return net, optimizer


def evaluation(inputs="./checkpoint_e30.pth.tar"):
    cudnn.benchmark = True
    cudnn.deterministic = True
    generator = model.ConvLSTMNetwork(10).cuda()
    generator, _ = load(generator, None, inputs)

    generator.eval()
    valid_folder = dataset.RadarDataset()
    valid = data.DataLoader(valid_folder, batch_size=1, shuffle=False, pin_memory=True, num_workers=15)
    with torch.no_grad():
        flag1 = 0
        flag5 = 0
        flag10 = 0
        flag15 = 0
        result1 = np.zeros([10, 5])
        result5 = np.zeros([10, 5])
        result10 = np.zeros([10, 5])
        result15 = np.zeros([10, 5])
        t = tqdm(valid, total=len(valid))
        for i, (targetVar, inputVar) in enumerate(t):
            inputs = inputVar.cuda()  # B,S,C,H,W
            label = targetVar.cuda()  # B,S,C,H,W
            generator_pred = generator(inputs)
            tl = torch.max(targetVar.view(-1))
            if tl >= 1.0 and flag1 < 100:
                for time in range(10):
                    result1[time, :] += np.array(acc(label[:, :time, ...], generator_pred[:, :time, ...], 1.0))
                flag1 += 1
            if tl >= 5.0 and flag5 < 100:
                for time in range(10):
                    result5[time, :] += np.array(acc(label[:, :time, ...], generator_pred[:, :time, ...], 5.0))
                flag5 += 1
            if tl >= 10.0 and flag10 < 100:
                for time in range(10):
                    result10[time, :] += np.array(acc(label[:, :time, ...], generator_pred[:, :time, ...], 10.0))
                flag10 += 1
            if tl >= 15.0 and flag15 < 100:
                for time in range(10):
                    result15[time, :] += np.array(acc(label[:, :time, ...], generator_pred[:, :time, ...], 15.0))
                flag15 += 1
            if flag1 >= 100 and flag5 >= 100 and flag10 >= 100 and flag15 >= 100:
                break

    return result1 / flag1, result5 / flag5, result10 / flag10, result15 / flag15


if __name__ == "__main__":
    res = evaluation()
    res1 = evaluation(inputs="checkpoint_e30_gan.pth.tar")
    np.save("conv.npy", res)
    np.save("conv-gan.npy", res1)
