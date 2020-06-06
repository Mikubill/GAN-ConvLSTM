import glob
import numpy as np
import time
import zarr
import torch
import torch.utils.data as data


class RadarDataset(data.Dataset):
    def __init__(self, train=True, threshold=None, n_frames_input=10, n_frames_output=10):
        """
        param num_objects: a list of number of possible objects.
        """
        super(RadarDataset, self).__init__()

        self.dataset = sorted(glob.glob("/home/mist/data/*"))
        self.train = train
        self.length = 500 if train else 50
        self.radar = [(583, 1840), (604, 2121), (727, 1835), (993, 1767), (1168, 1831), (1233, 1665), (1427, 1615),
                      (1456, 1756), (1590, 1610), (1539, 1517), (1411, 1451), (1606, 1412), (1494, 1208), (1647, 1167),
                      (1769, 1294), (1747, 988), (2083, 1038), (2352, 924), (2621, 781), (2808, 494)]
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.threshold = threshold

    def correcter(self, ones=True):
        inputs, output = self.getitem()

        while np.max(inputs) == 0:
            # print(np.mean(np.sum(inputs, (1, 2)), axis=0))
            # print(np.max(d, left, right))
            inputs, output = self.getitem()

        if ones:
            output = torch.from_numpy(output).contiguous().float().unsqueeze(1)
            inputs = torch.from_numpy(inputs).contiguous().float().unsqueeze(1)

        return output, inputs

    def getitem(self):
        rand = np.random.RandomState(round((time.time() - 1589500000) * 1000))
        now = rand.choice(self.dataset)
        index = self.dataset.index(now)
        if index < 20:
            ranger = self.dataset[index:index + self.n_frames_total]
        elif index > len(self.dataset) - 20:
            ranger = self.dataset[index - self.n_frames_total:index]
        else:
            ranger = self.dataset[index - self.n_frames_input:index + self.n_frames_output]

        full = self.get_content(ranger)
        inputs = full[:self.n_frames_input, ...]
        output = full[self.n_frames_input:self.n_frames_total, ...]
        return inputs, output

    def get_content(self, files):
        dat = []

        for item in files:
            band = np.fromfile(item, dtype='float32', sep='').reshape(3360, 2560)
            band[band == 9.999e+20] = 0
            dat.append(band)

        dataset = np.asarray(dat)
        return self.radar_selector(dataset)

    def radar_selector(self, dataset):
        rand = np.random.RandomState(round((time.time() - 1589500000) * 1000))
        crop1 = dataset[:, ..., :512, :512]
        crop2 = [crop1]
        for radar in self.radar:
            dx, dy = radar
            crop = dataset[:, ..., dx - 256:dx + 256, dy - 256:dy + 256]
            if np.mean(np.sum(np.reshape(crop, (crop.shape[0], -1)), axis=1)) > 200:
                crop2.append(crop)
        if len(crop2) == 1:
            return crop1
        else:
            return crop2[rand.randint(0, len(crop2) - 1)]

    def __getitem__(self, idx):
        return self.correcter()

    def __len__(self):
        return self.length


class RadarAndSatelliteDataset(RadarDataset):
    def __init__(self, **kw):
        super(RadarAndSatelliteDataset, self).__init__(**kw)
        from numcodecs import blosc
        blosc.set_nthreads(64)

        self.dataset_1 = [zarr.open("/home/mist/hmr-data/data-merged-201906.zarr", "r"),
                          zarr.open("/home/mist/hmr-data/data-merged-201907.zarr", "r")]
        self.dataset_2 = zarr.open("/home/mist/hmr-data/data-merged-201908.zarr", "r")

    def getitem(self):
        rand = np.random.RandomState(round((time.time() - 1589500000) * 1000))
        dataset = self.dataset_1[rand.choice([0, 1])]

        pos = rand.randint(dataset.shape[0])
        dat = dataset if self.train else self.dataset_2
        if pos < 20:
            full = dat[pos:pos + self.n_frames_total, ...]
        elif pos > dat.shape[0] - 20:
            full = dat[pos - self.n_frames_total:pos, ...]
        else:
            full = dat[pos - self.n_frames_input:pos + self.n_frames_output, ...]

        full = self.radar_selector(full)
        full[full == 9.999e+20] = 0
        full[:, 1, ...] = 300 - full[:, 1, ...]
        inputs = full[:self.n_frames_input, ...]
        output = full[self.n_frames_input:self.n_frames_total, ...]
        return inputs, output

    def correcter(self, ones=True):
        inputs, output = self.getitem()

        while np.mean(np.sum(inputs[:, 0, ...], (-1, -2)), axis=0) < 100 or inputs.shape != \
                (10, 2, 512, 512) or output.shape != (10, 2, 512, 512):
            # print(inputs.shape, output.shape)
            inputs, output = self.getitem()

        if ones:
            output = torch.from_numpy(output).contiguous().float()
            inputs = torch.from_numpy(inputs).contiguous().float()

        return output[:, 0, ...].reshape(10, 1, 512, 512), inputs
