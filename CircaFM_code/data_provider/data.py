import numpy as np
import math
import copy

def load_from_tsfile(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    return_meta_data=False,
    return_type="auto",
    Realcase=False
):
    if not full_file_path_and_name.endswith(".ts"):
        full_file_path_and_name = full_file_path_and_name + ".ts"
    with open(full_file_path_and_name, "r", encoding="utf-8") as file:
        meta_data = _load_header_info(file)
        data, y, meta_data = _load_data(file, meta_data, Realcase)

        dup_count = int(meta_data["dup"])
        if dup_count > 1:
            data = _process_replicate_data(data, y, meta_data, Realcase)
        else:
            if data.ndim == 3 and data.shape[1] > 1:
                data = np.mean(data, axis=1, keepdims=True)

    if meta_data["equallength"]:
        data = np.array(data)
        if return_type == "numpy2D" and meta_data["univariate"] and int(meta_data["dup"]) == 1:
            data = data.squeeze()

    if meta_data["targetlabel"]:
        y = y.astype(float)

    time_stamp = _get_timestamps_info(meta_data['timestamps'].copy(),
                                    data_shape=data.shape,
                                    dup_count=int(meta_data["dup"])) if meta_data['timestamps'] else meta_data['timestamps']

    return (data, y, time_stamp, meta_data) if return_meta_data else (data, y, time_stamp)


def _process_replicate_data(data, y, meta_data, Realcase):
    dup_count = int(meta_data["dup"])
    n_cases, n_channels, series_length = data.shape

    if n_channels != dup_count:
        raise ValueError(f"Channel count {n_channels} does not match dup count {dup_count}")

    concat_data = []
    for i in range(n_cases):
        sample_data = data[i]
        concatenated_series = sample_data.reshape(1, -1)
        concat_data.append(concatenated_series)

    concat_data = np.array(concat_data)
    return concat_data


def _get_timestamps_info(timestamps, data_shape, dup_count=1):
    n_cases, n_channels, total_length = data_shape

    if dup_count > 1:
        extended_timestamps = []
        for dup_idx in range(dup_count):
            extended_timestamps.extend(timestamps)

        time_stamp_array = np.array([[math.floor(t), (t - math.floor(t)) * 60] for t in extended_timestamps])
        time_stamp = np.tile(time_stamp_array, (n_cases, 1, 1))
    else:
        time_stamp = np.array([[math.floor(t), (t - math.floor(t)) * 60] for t in timestamps])
        time_stamp = np.tile(time_stamp, (n_cases, 1, 1))

    return time_stamp


def _load_header_info(file):
    meta_data = {
        "problemname": "none",
        "timestamps": [],
        "missing": False,
        "univariate": True,
        "dup": 1,
        "equallength": True,
        "classlabel": True,
        "targetlabel": False,
        "class_values": [],
    }
    boolean_keys = ["missing", "univariate", "equallength", "targetlabel"]
    for line in file:
        line = line.strip().lower()
        if line and not line.startswith("#"):
            tokens = line.split(" ")
            token_len = len(tokens)
            key = tokens[0][1:]
            if key == "data":
                if line != "@data":
                    raise IOError("data tag should not have an associated value")
                return meta_data

            if key in meta_data.keys():
                if key in boolean_keys:
                    if token_len != 2:
                        raise IOError(f"{tokens[0]} tag requires a boolean value")
                    if tokens[1] == "true":
                        meta_data[key] = True
                    elif tokens[1] == "false":
                        meta_data[key] = False
                elif key == "problemname" or key == "dup":
                    meta_data[key] = tokens[1]
                elif key == "timestamps":
                    meta_data[key] = [float(part) for part in tokens[1].split(',')] if tokens[1] != "false" else False
                elif key == "classlabel":
                    if tokens[1] == "true":
                        meta_data["classlabel"] = True
                        if token_len == 2:
                            raise IOError(
                                "if the classlabel tag is true then class values "
                                "must be supplied"
                            )
                    elif tokens[1] == "false":
                        meta_data["classlabel"] = False
                    else:
                        raise IOError("invalid class label value")
                    meta_data["class_values"] = [token.strip() for token in tokens[2:]]
        if meta_data["targetlabel"]:
            meta_data["classlabel"] = False
    return meta_data


def _load_data(file, meta_data, Realcase, replace_missing_vals_with="NaN"):
    data = []
    n_cases = 0
    n_channels = 0
    current_channels = 0
    series_length = 0
    y_values = []
    for line in file:
        line = line.strip().lower()
        line = line.replace("?", replace_missing_vals_with)
        channels = line.split(":")
        n_cases += 1
        current_channels = len(channels)
        if meta_data["classlabel"] or meta_data["targetlabel"]:
            current_channels -= 1
        if n_cases == 1:
            n_channels = current_channels
            if meta_data["equallength"]:
                series_length = len(channels[0].split(","))
        else:
            if current_channels != n_channels:
                raise IOError(
                    f"Inconsistent number of dimensions in case {n_cases}. "
                    f"Expecting {n_channels} but have read {current_channels}"
                )
            if meta_data["univariate"] and int(meta_data["dup"]) == 1 and current_channels > 1:
                raise IOError(
                    f"Seen {current_channels} in case {n_cases}."
                    f"Expecting univariate from meta data"
                )
        if meta_data["equallength"]:
            current_length = series_length
        else:
            current_length = len(channels[0].split(","))
        np_case = np.zeros(shape=(n_channels, current_length))
        for i in range(0, n_channels):
            single_channel = channels[i].strip()
            data_series = single_channel.split(",")
            data_series = [float(x) for x in data_series]
            if len(data_series) != current_length:
                raise IOError(
                    f"Unequal length series, in case {n_cases} meta "
                    f"data specifies all equal {series_length} but saw "
                    f"{len(single_channel)}"
                )
            np_case[i] = np.array(data_series)
        data.append(np_case)
        if meta_data["classlabel"] or meta_data["targetlabel"]:
            temp_label = channels[n_channels]
            if Realcase:
                temp_label = [x for x in channels[n_channels].split(",")]
            y_values.append(temp_label)
    if meta_data["equallength"]:
        data = np.array(data)

    return data, np.asarray(y_values), meta_data


def MultiGroup_load_from_tsfile(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    return_meta_data=False,
    return_type="auto",
    Realcase=False
):
    if not full_file_path_and_name.endswith(".ts"):
        full_file_path_and_name += ".ts"

    with open(full_file_path_and_name, "r", encoding="utf-8") as file:
        meta_data_1, meta_data_2 = _load_header_info_mulGroup(file)
        series_1, meta_data_1, series_2, meta_data_2, y = _load_data_mulGroup(
            file, meta_data_1, meta_data_2, Realcase
        )

        if meta_data_1["equallength"]:
            series_1 = np.array(series_1)
        if meta_data_2["equallength"]:
            series_2 = np.array(series_2)

        dup_count = int(meta_data_1["dup"])
        if dup_count > 1:
            series_1 = _process_replicate_data_multiGroup(series_1, dup_count, group_name="group1")
            series_2 = _process_replicate_data_multiGroup(series_2, dup_count, group_name="group2")

    if meta_data_1["equallength"] and return_type == "numpy2D" and meta_data_1["univariate"] and dup_count == 1:
        series_1 = series_1.squeeze()
    if meta_data_2["equallength"] and return_type == "numpy2D" and meta_data_2["univariate"] and dup_count == 1:
        series_2 = series_2.squeeze()

    if meta_data_1["targetlabel"] and meta_data_2["targetlabel"]:
        y = y.astype(float)

    time_stamp_1 = _get_timestamps_info_mulGroup_dup(
        meta_data_1['timestamps'].copy(),
        data_shape=series_1.shape,
        dup_count=dup_count
    ) if meta_data_1['timestamps'] else None

    time_stamp_2 = _get_timestamps_info_mulGroup_dup(
        meta_data_2['timestamps'].copy(),
        data_shape=series_2.shape,
        dup_count=dup_count
    ) if meta_data_2['timestamps'] else None

    return (series_1, time_stamp_1, series_2, time_stamp_2, y, meta_data_1, meta_data_2) if return_meta_data else \
           (series_1, time_stamp_1, series_2, time_stamp_2, y)


def _process_replicate_data_multiGroup(series_data, dup_count, group_name):
    n_cases, n_channels, series_length = series_data.shape

    if n_channels != dup_count:
        raise ValueError(f"{group_name} channel count {n_channels} does not match dup count {dup_count}")

    concat_data = []
    for i in range(n_cases):
        concatenated = series_data[i].reshape(1, -1)
        concat_data.append(concatenated)

    concat_data = np.array(concat_data)
    return concat_data


def _get_timestamps_info_mulGroup_dup(timestamps, data_shape, dup_count):
    n_cases, _, total_length = data_shape

    if dup_count > 1:
        extended_timestamps = []
        for _ in range(dup_count):
            extended_timestamps.extend(timestamps)

        time_stamp_array = np.array([
            [math.floor(t), (t - math.floor(t)) * 60]
            for t in extended_timestamps
        ])
        time_stamp = np.tile(time_stamp_array, (n_cases, 1, 1))
    else:
        time_stamp_array = np.array([
            [math.floor(t), (t - math.floor(t)) * 60]
            for t in timestamps
        ])
        time_stamp = np.tile(time_stamp_array, (n_cases, 1, 1))

    return time_stamp


def _load_header_info_mulGroup(file):
    meta_data_1 = {
        "problemname": "none",
        "timestamps": [],
        "missing": False,
        "univariate": True,
        "dup": 1,
        "equallength": True,
        "classlabel": True,
        "targetlabel": False,
        "class_values": [],
    }
    meta_data_2 = copy.deepcopy(meta_data_1)

    boolean_keys = ["missing", "univariate", "equallength", "targetlabel"]
    for line in file:
        line = line.strip().lower()
        if line and not line.startswith("#"):
            tokens = line.split(" ")
            token_len = len(tokens)
            key = tokens[0][1:]
            if key == "data":
                if line != "@data":
                    raise IOError("data tag should not have an associated value")
                break
            if key in meta_data_1.keys():
                if key in boolean_keys:
                    if token_len != 2:
                        raise IOError(f"{tokens[0]} tag requires a boolean value")
                    val = tokens[1] == "true"
                    meta_data_1[key] = val
                    meta_data_2[key] = val
                elif key in ["problemname", "dup"]:
                    meta_data_1[key] = tokens[1]
                    meta_data_2[key] = tokens[1]
                elif key == "timestamps":
                    if ";" in tokens[1]:
                        t1, t2 = tokens[1].split(';')
                        meta_data_1[key] = [float(x) for x in t1.split(',')] if t1 != "false" else False
                        meta_data_2[key] = [float(x) for x in t2.split(',')] if t2 != "false" else False
                elif key == "classlabel":
                    if tokens[1] == "true":
                        meta_data_1["classlabel"] = meta_data_2["classlabel"] = True
                        if token_len == 2:
                            raise IOError("classlabel true requires class values")
                    else:
                        meta_data_1["classlabel"] = meta_data_2["classlabel"] = False
                    meta_data_1["class_values"] = meta_data_2["class_values"] = [x.strip() for x in tokens[2:]]
        if meta_data_1["targetlabel"]:
            meta_data_1["classlabel"] = meta_data_2["classlabel"] = False

    return meta_data_1, meta_data_2


def _load_data_mulGroup(file, meta_data_1, meta_data_2, Realcase, replace_missing_vals_with="NaN"):
    series_1, series_2 = [], []
    y_values = []
    n_cases = 0
    n_channels_1 = n_channels_2 = 0
    series_length_1 = series_length_2 = 0

    for line in file:
        line = line.strip().replace("?", replace_missing_vals_with)
        data_1, data_2, label = line.split(";")

        if Realcase:
            parts = label.split(",")
            label = [parts[0]] + [int(x) for x in parts[1:]]
        else:
            label = [int(x) for x in label.split(",")]

        channels_1 = data_1.split(":")
        channels_2 = data_2.split(":")
        n_cases += 1
        curr_ch1, curr_ch2 = len(channels_1), len(channels_2)

        if n_cases == 1:
            n_channels_1, n_channels_2 = curr_ch1, curr_ch2
            if meta_data_1["equallength"]:
                series_length_1 = len(channels_1[0].split(","))
            if meta_data_2["equallength"]:
                series_length_2 = len(channels_2[0].split(","))
        else:
            if curr_ch1 != n_channels_1 or curr_ch2 != n_channels_2:
                raise IOError(f"Inconsistent channel count at case {n_cases}")

        curr_len1 = series_length_1 if meta_data_1["equallength"] else len(channels_1[0].split(","))
        np_case1 = np.zeros((n_channels_1, curr_len1))
        for i in range(n_channels_1):
            series = [float(x) for x in channels_1[i].split(",")]
            if len(series) != curr_len1:
                raise IOError(f"Length mismatch at case {n_cases}")
            np_case1[i] = series
        series_1.append(np_case1)

        curr_len2 = series_length_2 if meta_data_2["equallength"] else len(channels_2[0].split(","))
        np_case2 = np.zeros((n_channels_2, curr_len2))
        for i in range(n_channels_2):
            series = [float(x) for x in channels_2[i].split(",")]
            if len(series) != curr_len2:
                raise IOError(f"Length mismatch at case {n_cases}")
            np_case2[i] = series
        series_2.append(np_case2)

        y_values.append(label)

    return series_1, meta_data_1, series_2, meta_data_2, np.asarray(y_values)