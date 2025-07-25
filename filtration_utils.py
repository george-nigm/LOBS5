import os, glob, re
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def with_mean_std(func):
    """
    Decorator to add mean line and ±1 std interval band to a figure
    generated by a plot function.
    """
    def wrapper(merged, all_series, x, hist_steps, gen_block,
                num_insertions, num_coolings,
                n_gen_msgs, midprice_step_size, *args, **kwargs):
        # Call original plotting function
        fig = func(merged, all_series, x, hist_steps, gen_block,
                   num_insertions, num_coolings,
                   n_gen_msgs, midprice_step_size, *args, **kwargs)
        # Compute mean and standard deviation
        mean_series = all_series.mean(axis=0)
        std_series = all_series.std(axis=0)
        # Add ±1 std band
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([mean_series + std_series,
                              (mean_series - std_series)[::-1]]),
            fill='toself', fillcolor='rgba(0,0,0,0.1)',
            line=dict(color='rgba(0,0,0,0)'), hoverinfo='skip',
            showlegend=True, name='±1 std'
        ))
        # Add mean line
        fig.add_trace(go.Scatter(
            x=x, y=mean_series, mode='lines',
            name='Mean', line=dict(color='black', width=4)
        ))
        return fig
    return wrapper

def summary_table(experiment_name):
    DATA_DIR = f"/app/data_saved/{experiment_name}/mid_price"
    pattern = os.path.join(DATA_DIR, "mid_price_batch_*_iter_*.npy")

    # 2) Собираем все пути
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No .npy files matching {pattern}")

    # 3) Разбираем имя, загружаем и формируем «плоские» записи
    rx = re.compile(r"mid_price_batch_\[([\d,\s]+)\]_iter_(\d+)\.npy$")
    records = []
    for f in files:
        m = rx.search(os.path.basename(f))
        if not m:
            continue
        rng = m.group(1).replace(" ", "")      # e.g. "3464,4169,4497,6855"
        itr = int(m.group(2))                  # iteration number
        arr = np.load(f)                       # shape (n_steps, batch_size)

        ids = list(map(int, rng.split(",")))
        # now split out each column into its own sample-stream:
        for col, sample_id in enumerate(ids):
            records.append({
                "id":        sample_id,
                "iteration": itr,
                "data":      arr[:, col]
            })

    # 4) Создаём DataFrame и сортируем
    df = pd.DataFrame.from_records(records, columns=["id","iteration","data"])
    df = df.sort_values(["id","iteration"]).reset_index(drop=True)

    # 5) Группируем по id и склеиваем временные ряды
    merged = (
        df.groupby("id", as_index=False)
        .agg(merged_data=("data", lambda s: np.concatenate(s.tolist())))
    )

    # 6) (опционально) превратить в чистые списки
    merged["merged_data"] = merged["merged_data"].apply(lambda a: a.tolist())
    return merged

def build_zero_padded_series(hist_msgs, n_gen_msgs, midprice_step_size, merged):
    # convert to “mid-price sample” units
    hist_steps   = hist_msgs // midprice_step_size       # 500
    gen_steps     = n_gen_msgs // midprice_step_size     # 50
    gen_block     = gen_steps + 1                        # 51


    # build zeroed series…
    # build zeroed series, padded with zeros to the max length
    all_series_raw = []
    for row in merged.itertuples(index=False):
        data = np.array(row.merged_data)
        zeroed = data - data[0]
        # zeroed = data
        all_series_raw.append(zeroed)
    # find the longest series
    max_len = max(arr.shape[0] for arr in all_series_raw)
    # pad each series at the end with zeros up to max_len
    all_series = np.vstack([
        np.pad(
            arr,
            pad_width=(0, max_len - arr.shape[0]),
            mode="constant",
            constant_values=0
        )
        for arr in all_series_raw
    ])
    # now build your x-axis to match
    x = np.arange(1, max_len + 1)
    return x, all_series

def plot_midprice_series_with_insertions(
    merged,
    all_series,
    x,
    hist_steps,
    gen_block,
    num_insertions,
    num_coolings,
    n_gen_msgs,
    midprice_step_size,
    height=800,
    width=1200,
):
    """
    Plots zeroed midprice series with insertion and cooling vertical lines.
    """
    fig = go.Figure()

    for row, arr0 in zip(merged.itertuples(False), all_series):
        fig.add_trace(go.Scatter(
            x=x, y=arr0, mode='lines',
            opacity=1.0, hoverinfo='skip',
            line=dict(width=1),
            name=f"id {row.id}"
        ))

    fig.add_vline(x=hist_steps, line=dict(color='blue', width=2, dash='dash'))

    events = np.arange(1, num_insertions + num_coolings + 1)
    positions = hist_steps + gen_block * events

    for pos in positions[:num_insertions]:
        fig.add_vline(x=pos, line=dict(color='red', width=2, dash='solid'))

    for pos in positions[num_insertions:]:
        fig.add_vline(x=pos, line=dict(color='red', width=2, dash='dash'))

    print("insertion positions:", positions[:num_insertions].tolist())
    print("cooling positions:  ", positions[num_insertions:].tolist())

    fig.add_hline(y=0, line=dict(color='black', width=2, dash='solid'),
                  annotation_text="0-line", annotation_position="bottom right")

    fig.update_layout(
        title="All midprice series (zeroed) with insertion/cooling lines",
        xaxis_title="Index", yaxis_title="Price – first_price",
        template="plotly_white", hovermode="x unified",
        height=height, width=width, margin={"b": 150}
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    info = (
        f"Midprice every {midprice_step_size} msgs;<br>"
        f"solid red = insertion (every {n_gen_msgs} msgs +1 step);<br>"
        f"dashed red = cooling (same spacing); blue = end of history."
    )
    fig.add_annotation(
        text=info, xref="paper", yref="paper",
        x=0, y=-0.225, showarrow=False, align="left"
    )

    return fig


def plot_midprice_series_with_mean_std(
    merged,
    all_series,
    x,
    hist_steps,
    gen_block,
    num_insertions,
    num_coolings,
    n_gen_msgs,
    midprice_step_size,
    height=800,
    width=1200,
):
    fig = plot_midprice_series_with_insertions(
        merged,
        all_series,
        x,
        hist_steps,
        gen_block,
        num_insertions,
        num_coolings,
        n_gen_msgs,
        midprice_step_size,
        height,
        width,
    )

    mean_series = all_series.mean(axis=0)
    std_series = all_series.std(axis=0)

    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([mean_series + std_series,
                          (mean_series - std_series)[::-1]]),
        fill='toself', fillcolor='rgba(0,0,0,0.3)',
        line=dict(color='rgba(0,0,0,0)'), hoverinfo='skip',
        showlegend=True, name='±1 std'
    ))

    fig.add_trace(go.Scatter(
        x=x, y=mean_series, mode='lines',
        name='Mean', line=dict(color='black', width=4)
    ))

    return fig, mean_series, std_series



def prepare_volatility_filtered_series(merged, hist_msgs, n_gen_msgs, midprice_step_size, volatility_cutoff=0.10):
    """
    Computes volatility metrics, filters out the most volatile series,
    builds zero-padded, zeroed series for aligned plotting.

    Args:
        merged (pd.DataFrame): Must contain 'id' and 'merged_data' columns.
        hist_msgs (int): Number of historical messages.
        n_gen_msgs (int): Number of generated messages per insertion.
        midprice_step_size (int): Step size for midprice sampling.
        volatility_cutoff (float): Fraction of most volatile series to drop (default 10%).

    Returns:
        x (np.ndarray): X-axis array aligned with all_series.
        all_series (np.ndarray): Zero-padded, zeroed midprice series array.
        merged (pd.DataFrame): Filtered merged dataframe.
        hist_steps (int), gen_block (int): Step parameters for plotting.
    """
    # Compute volatility metrics
    merged["std_dev"] = merged["merged_data"].apply(lambda x: np.std(x))
    merged["max_abs_dev"] = merged["merged_data"].apply(lambda x: np.max(np.abs(np.array(x) - x[0])))

    # Sort and filter out the top volatility_cutoff fraction
    most_volatile = merged.sort_values(by="max_abs_dev", ascending=False).reset_index()
    print(most_volatile[["id", "std_dev", "max_abs_dev"]])

    n_before = len(most_volatile)
    print(f"\nBefore filtering: {n_before} samples")

    top_n = int(len(most_volatile) * volatility_cutoff)
    most_volatile = most_volatile[top_n:]
    merged = merged[merged['id'].isin(most_volatile['id'].values)].reset_index(drop=True)

    n_after = len(merged)
    print(f"\nAfter filtering: {n_after} samples")

    # Compute step parameters
    hist_steps = hist_msgs // midprice_step_size
    gen_steps = n_gen_msgs // midprice_step_size
    gen_block = gen_steps + 1

    # Build zeroed, zero-padded series
    all_series_raw = []
    for row in merged.itertuples(index=False):
        data = np.array(row.merged_data)
        zeroed = data - data[0]
        all_series_raw.append(zeroed)

    max_len = max(arr.shape[0] for arr in all_series_raw)
    all_series = np.vstack([
        np.pad(arr, pad_width=(0, max_len - arr.shape[0]), mode="constant", constant_values=0)
        for arr in all_series_raw
    ])
    x = np.arange(1, max_len + 1)

    return x, all_series, merged, hist_steps, gen_block

# add check if the right book state is in the player?