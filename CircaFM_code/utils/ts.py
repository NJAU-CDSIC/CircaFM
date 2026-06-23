import pandas as pd
from collections import defaultdict
import numpy as np

def singal_convert_to_ts(input_csv, output_ts, problem_name, label_col='label'):
    df = pd.read_csv(input_csv, index_col=0)
    
    if label_col in df.columns:
        labels = df.pop(label_col)
    else:
        raise ValueError(f"Label column '{label_col}' not found")

    time_point_order = []
    rep_groups = defaultdict(list)

    for col in df.columns:
        parts = col.split('_')
        if len(parts) == 2:
            try:
                time_point = float(parts[0])
                rep_num = int(parts[1])
            except ValueError:
                raise ValueError(f"Cannot parse column '{col}' as time_replicate (time can be float, replicate must be integer)")
        else:
            raise ValueError(f"Expected column format 'time_replicate', got: '{col}'")
        
        if time_point not in time_point_order:
            time_point_order.append(time_point)
        rep_groups[rep_num].append((time_point, col))

    if not rep_groups:
        raise ValueError("No replicate groups found, check column names")

    time_point_order.sort()
    num_time_points = len(time_point_order)
    num_reps = len(rep_groups)
    
    for rep_num, cols in rep_groups.items():
        if len(cols) != num_time_points:
            raise ValueError(f"Replicate group {rep_num} has inconsistent time points")

    equal_length = True
    if num_time_points > 2:
        diffs = np.diff(time_point_order)
        first_diff = diffs[0]
        for d in diffs[1:]:
            if not np.isclose(d, first_diff, rtol=1e-5, atol=1e-8):
                equal_length = False
                break
    elif num_time_points <= 1:
        equal_length = True

    time_stamps_str = ','.join([str(tp) for tp in time_point_order])
    meta_info = [
        f"#{problem_name} provenance not determined yet",
        f"@problemName {problem_name}",
        f"@timeStamps {time_stamps_str}",
        "@missing false",
        "@univariate true",
        f"@dup {num_reps}",
        f"@equalLength {str(equal_length).lower()}",
        f"@seriesLength {num_time_points * num_reps}",
        "@classLabel true 0 1",
        "@data"
    ]

    ts_lines = meta_info.copy()
    for gene_id in df.index:
        group_values = []
        for rep_num in sorted(rep_groups.keys()):
            current_group = rep_groups[rep_num]
            tp_to_col = dict(current_group)
            current_values = [str(df.loc[gene_id, tp_to_col[tp]]) for tp in time_point_order]
            group_values.append(','.join(current_values))
        values_str = ':'.join(group_values)
        final_line = f"{values_str}:{gene_id},{int(labels.loc[gene_id])}"
        ts_lines.append(final_line)

    with open(output_ts, 'w') as f:
        f.write('\n'.join(ts_lines))
    print(f"TS file generated: {output_ts}")


def multi_convert_to_ts(input_csv, output_ts, problem_name, label_cols):
    df = pd.read_csv(input_csv, index_col=0)

    if all(col in df.columns for col in label_cols):
        labels = df[label_cols].copy()
        df = df.drop(columns=label_cols)
    else:
        missing = [col for col in label_cols if col not in df.columns]
        raise ValueError(f"Label columns {missing} not found")

    time_point_order_1 = []
    time_point_order_2 = []
    rep_groups_1 = defaultdict(list)
    rep_groups_2 = defaultdict(list)

    for col in df.columns:
        parts = col.split('_')
        if len(parts) != 3:
            raise ValueError(f"Expected column format 'time_replicate_series', got: {col}")
        try:
            time_point = float(parts[0])
            rep_num = int(parts[1])
            ts_idx = int(parts[2])
        except ValueError:
            raise ValueError(f"Column parse error: {col}, time point must be numeric, replicate/series must be integer")

        if ts_idx == 0:
            if time_point not in time_point_order_1:
                time_point_order_1.append(time_point)
            rep_groups_1[rep_num].append((time_point, col))
        else:
            if time_point not in time_point_order_2:
                time_point_order_2.append(time_point)
            rep_groups_2[rep_num].append((time_point, col))

    time_point_order_1.sort()
    time_point_order_2.sort()

    if not rep_groups_1 or not rep_groups_2:
        raise ValueError("No data found, check column names")

    num_reps_1 = len(rep_groups_1)
    num_reps_2 = len(rep_groups_2)
    num_time_points_1 = len(time_point_order_1)
    num_time_points_2 = len(time_point_order_2)

    for rep_num, cols in rep_groups_1.items():
        if len(cols) != num_time_points_1:
            raise ValueError(f"Inconsistent time points in series 1 replicate {rep_num}")
    for rep_num, cols in rep_groups_2.items():
        if len(cols) != num_time_points_2:
            raise ValueError(f"Inconsistent time points in series 2 replicate {rep_num}")

    def check_equal_length(tp_list):
        if len(tp_list) <= 1:
            return True
        diffs = [tp_list[i+1] - tp_list[i] for i in range(len(tp_list)-1)]
        first_diff = diffs[0]
        for d in diffs[1:]:
            if abs(d - first_diff) > 1e-6:
                return False
        return True

    equal_len_1 = check_equal_length(time_point_order_1)
    equal_len_2 = check_equal_length(time_point_order_2)

    ts1_str = ','.join(str(tp) for tp in time_point_order_1)
    ts2_str = ','.join(str(tp) for tp in time_point_order_2)

    meta_info = [
        f"#{problem_name} provenance not determined yet",
        f"@problemName {problem_name}",
        f"@timeStamps {ts1_str};{ts2_str}",
        "@missing false",
        "@univariate false",
        f"@dup {num_reps_1}",
        f"@equalLength {str(equal_len_1).lower()}",
        f"@seriesLength {num_time_points_1 * num_reps_1}",
        "@classLabel true -1 0 1",
        "@data"
    ]

    ts_lines = meta_info.copy()
    for gene_id in df.index:
        line_parts = []
        group_values_1 = []
        for rep_num in sorted(rep_groups_1.keys()):
            current_group = rep_groups_1[rep_num]
            tp_to_col = dict(current_group)
            vals = [str(df.loc[gene_id, tp_to_col[tp]]) for tp in time_point_order_1]
            group_values_1.append(','.join(vals))
        line_parts.append(':'.join(group_values_1))

        group_values_2 = []
        for rep_num in sorted(rep_groups_2.keys()):
            current_group = rep_groups_2[rep_num]
            tp_to_col = dict(current_group)
            vals = [str(df.loc[gene_id, tp_to_col[tp]]) for tp in time_point_order_2]
            group_values_2.append(','.join(vals))
        line_parts.append(':'.join(group_values_2))

        data_str = ';'.join(line_parts)
        label_str = ','.join(str(int(labels.loc[gene_id, col])) for col in label_cols)
        final_line = f"{data_str};{gene_id},{label_str}"
        ts_lines.append(final_line)

    with open(output_ts, 'w') as f:
        f.write('\n'.join(ts_lines))
    print(f"Multi time-series TS file generated: {output_ts}")