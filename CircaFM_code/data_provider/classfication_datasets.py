import os
import numpy as np
from sklearn.preprocessing import StandardScaler

from data_provider.data import load_from_tsfile, MultiGroup_load_from_tsfile


class ClassificationDataset:

    def __init__(self, data_split="train", file_path="datasets/", seq_len=72):
        self.seq_len = seq_len
        self.train_file_path_and_name = os.path.join(file_path, "TRAIN.ts")
        self.val_file_path_and_name = os.path.join(file_path, "VALIDATION.ts")
        self.test_file_path_and_name = os.path.join(file_path, "TEST.ts")
        self.data_split = data_split

        self._read_data()

    @staticmethod
    def _transform_labels(train_labels, val_labels, test_labels):
        unique_labels = np.unique(train_labels)
        mapping = {label: idx for idx, label in enumerate(unique_labels)}
        train_labels = np.vectorize(mapping.get)(train_labels)
        val_labels = np.vectorize(mapping.get)(val_labels)
        test_labels = np.vectorize(mapping.get)(test_labels)
        return train_labels, val_labels, test_labels

    def __len__(self):
        return self.num_timeseries

    def _read_data(self):
        self.scaler = StandardScaler()

        self.train_data, self.train_labels = load_from_tsfile(self.train_file_path_and_name)
        self.val_data, self.val_labels = load_from_tsfile(self.val_file_path_and_name)
        self.test_data, self.test_labels = load_from_tsfile(self.test_file_path_and_name)

        self.train_labels, self.val_labels, self.test_labels = self._transform_labels(
            self.train_labels, self.val_labels, self.test_labels
        )

        if self.data_split == "train":
            self.data = self.train_data
            self.labels = self.train_labels
        elif self.data_split == "val":
            self.data = self.val_data
            self.labels = self.val_labels
        else:
            self.data = self.test_data
            self.labels = self.test_labels

        self.num_timeseries = self.data.shape[0]
        self.len_timeseries = self.data.shape[2]

        self.data = self.data.reshape(-1, self.len_timeseries)
        self.scaler.fit(self.data)
        self.data = self.scaler.transform(self.data)
        self.data = self.data.reshape(self.num_timeseries, self.len_timeseries)
        self.data = self.data.T

    def __getitem__(self, index):
        assert index < self.__len__()
        timeseries = self.data[:, index]
        timeseries_len = len(timeseries)
        labels = self.labels[index].astype(int)

        input_mask = np.ones(self.seq_len)
        input_mask[: self.seq_len - timeseries_len] = 0

        timeseries = np.pad(timeseries, (self.seq_len - timeseries_len, 0))
        return np.expand_dims(timeseries, axis=0), input_mask, labels


class DataSplit:

    def __init__(self, data, labels, time_stamp, seq_len=72):
        self.data = data
        self.labels = labels
        self.time_stamp = time_stamp
        self.seq_len = seq_len
        self._length = len(self.data)
        self.timesteps = self.data.shape[-1]

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        assert index < self.__len__()
        labels = self.labels[index].astype(int)
        timeseries = self.data[index]
        timeseries_len = timeseries.shape[-1]

        input_mask = np.ones(self.seq_len)
        input_mask[: self.seq_len - timeseries_len] = 0

        timeseries = np.pad(timeseries, ((0, 0), (self.seq_len - self.timesteps, 0)))
        x_mark = np.pad(
            self.time_stamp,
            ((0, 0), (self.seq_len - self.timesteps, 0), (0, 0)),
            constant_values=0,
        )
        return timeseries, input_mask, x_mark[index], labels


class newClassificationDataset:

    def __init__(self, data_split=None, file_path="datasets/", seq_len=72, Realcase=False):
        self.Realcase = Realcase
        self.seq_len = seq_len
        self.train_file_path_and_name = os.path.join(file_path, "TRAIN.ts")
        self.val_file_path_and_name = os.path.join(file_path, "VALIDATION.ts")
        self.test_file_path_and_name = os.path.join(file_path, "TEST.ts")
        self.data_split = data_split
        self.singleDatapath = os.path.join(file_path, f"{data_split.upper()}.ts") if data_split else None

        self._read_data()

    @staticmethod
    def _transform_labels(train_labels, val_labels, test_labels):
        unique_labels = np.unique(train_labels)
        mapping = {label: idx for idx, label in enumerate(unique_labels)}
        train_labels = np.vectorize(mapping.get)(train_labels)
        val_labels = np.vectorize(mapping.get)(val_labels)
        test_labels = np.vectorize(mapping.get)(test_labels)
        return train_labels, val_labels, test_labels

    @property
    def train_dat(self):
        return self._get_data("train")

    @property
    def val_dat(self):
        return self._get_data("val")

    @property
    def test_dat(self):
        return self._get_data("test")

    @property
    def aper_dat(self):
        return self._get_data("aper")

    def _read_data(self):
        if self.data_split:
            self.singleData, self.aperLabels, self.aper_time_stamp = load_from_tsfile(
                self.singleDatapath, return_meta_data=False, Realcase=self.Realcase
            )
            self.singleData = self._process_data(self.singleData)
        else:
            self.train_data, self.train_labels, self.train_time_stamp = load_from_tsfile(
                self.train_file_path_and_name, return_meta_data=False, Realcase=self.Realcase
            )
            self.val_data, self.val_labels, self.val_time_stamp = load_from_tsfile(
                self.val_file_path_and_name, return_meta_data=False, Realcase=self.Realcase
            )
            self.test_data, self.test_labels, self.test_time_stamp = load_from_tsfile(
                self.test_file_path_and_name, return_meta_data=False, Realcase=self.Realcase
            )

            if not self.Realcase:
                self.train_labels, self.val_labels, self.test_labels = self._transform_labels(
                    self.train_labels, self.val_labels, self.test_labels
                )

            self.train_data = self._process_data(self.train_data)
            self.val_data = self._process_data(self.val_data)
            self.test_data = self._process_data(self.test_data)

    @staticmethod
    def _process_data(data):
        n_samples, n_channels, timesteps = data.shape
        data = data.reshape(n_samples * n_channels, timesteps)
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)
        data = (data - mean) / (std + 1e-6)
        data = data.reshape(n_samples, n_channels, timesteps)
        return data

    def _get_data(self, split):
        if split == "aper":
            data, labels, time_stamp = self.singleData, self.aperLabels, self.aper_time_stamp
        elif split == "train":
            data, labels, time_stamp = self.train_data, self.train_labels, self.train_time_stamp
        elif split == "val":
            data, labels, time_stamp = self.val_data, self.val_labels, self.val_time_stamp
        elif split == "test":
            data, labels, time_stamp = self.test_data, self.test_labels, self.test_time_stamp
        else:
            raise ValueError(f"Unknown split: {split}")

        return DataSplit(data, labels, time_stamp, self.seq_len)


class MultipleDataset(DataSplit):

    def __init__(self, data_split="train", file_paths=None, seq_len=72, seed=123, Realcase=False):
        if file_paths is None:
            file_paths = ["datasets/"]
        self.data_split = data_split
        self.datasets = []
        self.seq_len = seq_len
        self.seed = seed
        self.mask = np.ones(self.seq_len)
        self.Realcase = Realcase

        aper_split = "train" if self.data_split == "aper" else None

        for file_path in file_paths:
            dataset = newClassificationDataset(
                file_path=file_path,
                seq_len=seq_len,
                data_split=aper_split,
                Realcase=self.Realcase,
            )
            self.datasets.append(dataset)

        self._merge_datasets()

    def _merge_datasets(self):
        all_data = []
        all_labels = []
        all_x_mark = []
        all_timesteps = []

        for dataset in self.datasets:
            if self.data_split == "aper":
                split_data = dataset.aper_dat
            elif self.data_split == "train":
                split_data = dataset.train_dat
            elif self.data_split == "val":
                split_data = dataset.val_dat
            else:
                split_data = dataset.test_dat

            num_samples = split_data.data.shape[0]
            all_timesteps.extend([split_data.timesteps] * num_samples)

            labels = split_data.labels
            data = split_data.data
            time_stamp = split_data.time_stamp

            x_mark = np.pad(
                time_stamp,
                ((0, 0), (self.seq_len - split_data.timesteps, 0), (0, 0)),
                constant_values=0,
            )
            data = np.pad(data, ((0, 0), (0, 0), (self.seq_len - split_data.timesteps, 0)))

            all_data.append(data)
            all_x_mark.append(x_mark)
            all_labels.append(labels)

        self.timesteps = split_data.timesteps
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_x_mark = np.concatenate(all_x_mark, axis=0)
        all_timesteps = np.array(all_timesteps)

        indices = np.arange(all_data.shape[0])
        np.random.seed(self.seed)
        np.random.shuffle(indices)

        self.data = all_data[indices]
        self.labels = all_labels[indices]
        self.x_mark = all_x_mark[indices]
        self.num_timeseries = self.data.shape[0]
        self.timesteps_per_sample = all_timesteps[indices]

    def __len__(self):
        return self.num_timeseries

    def __getitem__(self, index):
        assert index < self.__len__()
        labels = self.labels[index]
        labels = labels.tolist()
        if isinstance(labels, (list, tuple)) and len(labels) >= 2:
            labels = [labels[0], int(labels[1])]
        else:
            labels = int(labels)

        timeseries = self.data[index]
        x_mark = self.x_mark[index]

        input_mask = np.ones(self.seq_len)
        current_timesteps = self.timesteps_per_sample[index]
        input_mask[: self.seq_len - current_timesteps] = 0

        return timeseries, input_mask, x_mark, labels


class MulGroup_DataSplit:

    def __init__(self, data_1, time_stamp_1, data_2, time_stamp_2, labels, seq_len=72,
                 sample_ids=None, sample_folders=None):
        self.data_1 = data_1
        self.data_2 = data_2
        self.time_stamp_1 = time_stamp_1
        self.time_stamp_2 = time_stamp_2
        self.labels = labels
        self.seq_len = seq_len
        self.sample_ids = sample_ids
        self.sample_folders = sample_folders

        self._length_1 = len(data_1)
        self._length_2 = len(data_2)
        self.timesteps_1 = data_1.shape[-1]
        self.timesteps_2 = data_2.shape[-1]

    def __len__(self):
        if self._length_1 == self._length_2:
            return self._length_1
        return -1

    def __getitem__(self, index):
        assert index < self.__len__()
        labels = self.labels[index].astype(int)
        ts1 = self.data_1[index]
        ts2 = self.data_2[index]

        input_mask_1 = np.ones(self.seq_len)
        input_mask_1[: self.seq_len - self.timesteps_1] = 0
        input_mask_2 = np.ones(self.seq_len)
        input_mask_2[: self.seq_len - self.timesteps_2] = 0

        ts1 = np.pad(ts1, ((0, 0), (self.seq_len - self.timesteps_1, 0)))
        ts2 = np.pad(ts2, ((0, 0), (self.seq_len - self.timesteps_2, 0)))
        xm1 = np.pad(self.time_stamp_1,
                     ((0, 0), (self.seq_len - self.timesteps_1, 0), (0, 0)),
                     constant_values=0)
        xm2 = np.pad(self.time_stamp_2,
                     ((0, 0), (self.seq_len - self.timesteps_2, 0), (0, 0)),
                     constant_values=0)

        sample_id = self.sample_ids[index]
        sample_folder = self.sample_folders[index]

        return (ts1, input_mask_1, xm1[index],
                ts2, input_mask_2, xm2[index],
                labels, sample_id, sample_folder)


class MulGroup_ClassificationDataset:

    def __init__(self, data_split=None, file_path="datasets/", seq_len=72, Realcase=False):
        self.seq_len = seq_len
        self.data_split = data_split
        self.Realcase = Realcase
        self.file_path = file_path
        self.train_path = os.path.join(file_path, "TRAIN.ts")
        self.val_path = os.path.join(file_path, "VALIDATION.ts")
        self.test_path = os.path.join(file_path, "TEST.ts")

        self._read_data()

    def _read_data(self):
        (self.train_data_1, self.train_time_stamp_1,
         self.train_data_2, self.train_time_stamp_2,
         self.train_labels_raw) = MultiGroup_load_from_tsfile(
            self.train_path, return_meta_data=False, Realcase=self.Realcase)
        (self.val_data_1, self.val_time_stamp_1,
         self.val_data_2, self.val_time_stamp_2,
         self.val_labels_raw) = MultiGroup_load_from_tsfile(
            self.val_path, return_meta_data=False, Realcase=self.Realcase)
        (self.test_data_1, self.test_time_stamp_1,
         self.test_data_2, self.test_time_stamp_2,
         self.test_labels_raw) = MultiGroup_load_from_tsfile(
            self.test_path, return_meta_data=False, Realcase=self.Realcase)

    def _get_data(self, split):
        if split == "train":
            d1, ts1, d2, ts2, labels_raw = (
                self.train_data_1, self.train_time_stamp_1,
                self.train_data_2, self.train_time_stamp_2,
                self.train_labels_raw)
        elif split == "val":
            d1, ts1, d2, ts2, labels_raw = (
                self.val_data_1, self.val_time_stamp_1,
                self.val_data_2, self.val_time_stamp_2,
                self.val_labels_raw)
        elif split == "test":
            d1, ts1, d2, ts2, labels_raw = (
                self.test_data_1, self.test_time_stamp_1,
                self.test_data_2, self.test_time_stamp_2,
                self.test_labels_raw)
        else:
            raise ValueError(f"Unknown split: {split}")

        sample_ids = labels_raw[:, 0].astype(str)
        labels = labels_raw[:, 1:]
        folder_name = os.path.basename(os.path.normpath(self.file_path))
        sample_folders = np.array([folder_name] * len(sample_ids))

        return MulGroup_DataSplit(
            data_1=d1, time_stamp_1=ts1,
            data_2=d2, time_stamp_2=ts2,
            labels=labels,
            seq_len=self.seq_len,
            sample_ids=sample_ids,
            sample_folders=sample_folders,
        )

    @property
    def train_dat(self):
        return self._get_data("train")

    @property
    def val_dat(self):
        return self._get_data("val")

    @property
    def test_dat(self):
        return self._get_data("test")


class MulGroup_MultipleDataset(MulGroup_DataSplit):

    def __init__(self, data_split="train", file_paths=None, seq_len=72,
                 seed=123, Realcase=False, normalize_method="separate"):
        if file_paths is None:
            file_paths = ["datasets/"]
        self.data_split = data_split
        self.datasets = []
        self.seq_len = seq_len
        self.seed = seed
        self.Realcase = Realcase
        self.normalize_method = normalize_method

        for file_path in file_paths:
            ds = MulGroup_ClassificationDataset(
                data_split=self.data_split,
                file_path=file_path,
                seq_len=seq_len,
                Realcase=self.Realcase,
            )
            self.datasets.append(ds)

        self._merge_datasets()

    def _merge_datasets(self):
        all_data_1, all_data_2 = [], []
        all_xm_1, all_xm_2 = [], []
        all_labels = []
        all_sample_ids = []
        all_sample_folders = []
        all_timesteps_1, all_timesteps_2 = [], []

        for ds in self.datasets:
            if self.data_split == "train":
                split_data = ds.train_dat
            elif self.data_split == "val":
                split_data = ds.val_dat
            else:
                split_data = ds.test_dat

            curr_ts1 = split_data.timesteps_1
            curr_ts2 = split_data.timesteps_2
            n_samples = split_data.data_1.shape[0]

            all_timesteps_1.extend([curr_ts1] * n_samples)
            all_timesteps_2.extend([curr_ts2] * n_samples)

            labels = split_data.labels
            d1, d2 = split_data.data_1, split_data.data_2
            ts1, ts2 = split_data.time_stamp_1, split_data.time_stamp_2

            xm1 = np.pad(ts1, ((0, 0), (self.seq_len - curr_ts1, 0), (0, 0)),
                         constant_values=0)
            xm2 = np.pad(ts2, ((0, 0), (self.seq_len - curr_ts2, 0), (0, 0)),
                         constant_values=0)
            d1 = np.pad(d1, ((0, 0), (0, 0), (self.seq_len - curr_ts1, 0)))
            d2 = np.pad(d2, ((0, 0), (0, 0), (self.seq_len - curr_ts2, 0)))

            all_data_1.append(d1)
            all_data_2.append(d2)
            all_xm_1.append(xm1)
            all_xm_2.append(xm2)
            all_labels.append(labels)
            all_sample_ids.append(split_data.sample_ids)
            all_sample_folders.append(split_data.sample_folders)

        self.data_1 = np.concatenate(all_data_1, axis=0)
        self.data_2 = np.concatenate(all_data_2, axis=0)
        self.time_stamp_1 = np.concatenate(all_xm_1, axis=0)
        self.time_stamp_2 = np.concatenate(all_xm_2, axis=0)
        self.labels = np.concatenate(all_labels, axis=0)
        self.sample_ids = np.concatenate(all_sample_ids, axis=0)
        self.sample_folders = np.concatenate(all_sample_folders, axis=0)
        all_timesteps_1 = np.array(all_timesteps_1)
        all_timesteps_2 = np.array(all_timesteps_2)

        self.timesteps_1 = self.data_1.shape[-1]
        self.timesteps_2 = self.data_2.shape[-1]

        indices = np.arange(self.data_1.shape[0])
        np.random.seed(self.seed)
        np.random.shuffle(indices)

        self.data_1 = self.data_1[indices]
        self.data_2 = self.data_2[indices]
        self.x_mark_1 = self.time_stamp_1[indices]
        self.x_mark_2 = self.time_stamp_2[indices]
        self.labels = self.labels[indices]
        self.sample_ids = self.sample_ids[indices]
        self.sample_folders = self.sample_folders[indices]
        self.timesteps_1_per_sample = all_timesteps_1[indices]
        self.timesteps_2_per_sample = all_timesteps_2[indices]
        self.num_timeseries_1 = self.data_1.shape[0]
        self.num_timeseries_2 = self.data_2.shape[0]

    def __len__(self):
        if self.num_timeseries_1 == self.num_timeseries_2:
            return self.num_timeseries_1
        return -1

    def _normalize_sequences_together(self, seq1, seq2, ts1_len, ts2_len):
        valid1 = seq1[:, -ts1_len:]
        valid2 = seq2[:, -ts2_len:]
        combined = np.concatenate([valid1, valid2], axis=-1)
        mean = np.mean(combined, axis=-1, keepdims=True)
        std = np.std(combined, axis=-1, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        normalized = (combined - mean) / std
        seq1_norm = normalized[:, :ts1_len]
        seq2_norm = normalized[:, ts1_len:]
        seq1_final = np.pad(seq1_norm, ((0, 0), (self.seq_len - ts1_len, 0)))
        seq2_final = np.pad(seq2_norm, ((0, 0), (self.seq_len - ts2_len, 0)))
        return seq1_final, seq2_final

    def _normalize_sequences_separate(self, seq1, seq2, ts1_len, ts2_len):
        valid1 = seq1[:, -ts1_len:]
        mean1 = np.mean(valid1, axis=-1, keepdims=True)
        std1 = np.std(valid1, axis=-1, keepdims=True)
        std1 = np.where(std1 == 0, 1.0, std1)
        seq1_norm = (valid1 - mean1) / std1

        valid2 = seq2[:, -ts2_len:]
        mean2 = np.mean(valid2, axis=-1, keepdims=True)
        std2 = np.std(valid2, axis=-1, keepdims=True)
        std2 = np.where(std2 == 0, 1.0, std2)
        seq2_norm = (valid2 - mean2) / std2

        seq1_final = np.pad(seq1_norm, ((0, 0), (self.seq_len - ts1_len, 0)))
        seq2_final = np.pad(seq2_norm, ((0, 0), (self.seq_len - ts2_len, 0)))
        return seq1_final, seq2_final

    def __getitem__(self, index):
        assert index < self.__len__()
        labels = self.labels[index].astype(int)
        ts1 = self.data_1[index]
        ts2 = self.data_2[index]
        xm1 = self.x_mark_1[index]
        xm2 = self.x_mark_2[index]

        curr_ts1 = self.timesteps_1_per_sample[index]
        curr_ts2 = self.timesteps_2_per_sample[index]

        if self.normalize_method == "together":
            ts1, ts2 = self._normalize_sequences_together(ts1, ts2, curr_ts1, curr_ts2)
        elif self.normalize_method == "separate":
            ts1, ts2 = self._normalize_sequences_separate(ts1, ts2, curr_ts1, curr_ts2)
        elif self.normalize_method in ("none", None):
            ts1[:, :self.seq_len - curr_ts1] = 0
            ts2[:, :self.seq_len - curr_ts2] = 0
        else:
            raise ValueError(f"Unknown normalization method: {self.normalize_method}")

        input_mask_1 = np.ones(self.seq_len)
        input_mask_1[: self.seq_len - curr_ts1] = 0
        input_mask_2 = np.ones(self.seq_len)
        input_mask_2[: self.seq_len - curr_ts2] = 0

        sample_id = self.sample_ids[index]
        sample_folder = self.sample_folders[index]

        return (ts1, input_mask_1, xm1,
                ts2, input_mask_2, xm2,
                labels, sample_id, sample_folder)