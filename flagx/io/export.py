
import os
import flowio
import numpy as np
import pandas as pd
import scanpy as sc

from typing import List, Tuple, Dict, Union


def export_to_fcs(
        data_list: List[sc.AnnData],
        layer_key: Union[str, None] = None,
        sample_wise: bool = False,
        add_columns: Union[List[List[np.ndarray]], None] = None,
        add_columns_names: Union[List[str], None] = None,
        scale_columns: Union[List[str], None] = None,
        val_range: Tuple[float, float] = (0.0, 2**20),
        keep_unscaled: bool = False,
        save_path: Union[str, None] = None,
        save_filenames: Union[str, List[str], None] = None,
):
    if add_columns is not None and add_columns_names is None:
        raise ValueError("'add_columns_names' must not be None if 'add_columns' is not None")

    if add_columns is not None and add_columns_names is not None:
        if len(add_columns) != len(add_columns_names):
            raise ValueError("'add_columns' and 'add_columns_names' must have the same length")

    # Initialize dataframes from AnnData
    fcs_dfs = _init_fcs_dfs(data_list=data_list, layer_key=layer_key)

    # Add columns with sample id
    if not sample_wise:

        # Check whether any of the dataframes already contains a sample id column
        exists_id_col = any('sample_id' in fcs_df.columns for fcs_df in fcs_dfs)

        # Only annotate with sample id if no previous annotations are found
        if not exists_id_col:

            sample_ids = []
            fns = []
            for i, adata in enumerate(data_list):
                fn = adata.uns['filename']
                fns.append(fn)
                sample_ids.append(np.full(adata.shape[0], i + 1))

            if add_columns is None:
                add_columns = [sample_ids, ]
                add_columns_names = ['sample_id', ]
            else:
                add_columns.append(sample_ids)
                add_columns_names.append('sample_id')

            sample_fn_id_df = pd.DataFrame({'filenames': fns, 'sample_id': range(1, len(fns) + 1)})
            sample_fn_id_df.to_csv(os.path.join(save_path, 'filenames_and_sample_id.csv'), index=False)

            if scale_columns is None:
                scale_columns = ['sample_id', ]
            else:
                scale_columns.append('sample_id')

    # Add additional columns to each sample's dataframe
    if add_columns is not None:

        for col_list, col_name in zip(add_columns, add_columns_names):
            for col, fcs_df in zip(col_list, fcs_dfs):
                fcs_df[col_name] = col

    # Scale selected columns
    if scale_columns is not None:
        for scale_col in scale_columns:
            x_cols = [fcs_df[scale_col].to_numpy() for fcs_df in fcs_dfs]
            x_cols_scaled = _scale_columns(cols=x_cols, val_range=val_range, sample_wise=sample_wise)

            for fcs_df, x_col_scaled in zip(fcs_dfs, x_cols_scaled):

                if keep_unscaled:
                    fcs_df[scale_col + '_unscaled'] = fcs_df[scale_col].copy()

                fcs_df[scale_col] = x_col_scaled

    _save_to_fcs(
        fcs_dfs=fcs_dfs,
        val_range=val_range,
        save_path=save_path,
        save_filenames=save_filenames,
        fcs_metadata_dicts=[dict(), ] * len(fcs_dfs) if sample_wise else dict(),
        sample_wise=sample_wise,
    )


def _init_fcs_dfs(
        data_list: List[sc.AnnData],
        layer_key: Union[str, None],
) -> List[pd.DataFrame]:

    fcs_dfs = []
    for adata in data_list:
        if layer_key is None:
            x = adata.X.copy()
        else:
            x = adata.layers[layer_key].copy()
        df = pd.DataFrame(x, columns=adata.var_names)
        fcs_dfs.append(df)

    return fcs_dfs


def _scale_columns(
        cols: List[np.ndarray],
        val_range: Tuple[float, float],
        sample_wise: bool,
) -> List[np.ndarray]:

    min_val, max_val = val_range
    val_scale = max_val - min_val

    margin = val_scale * 0.05
    min_val = min_val + margin
    max_val = max_val - margin
    val_scale = max_val - min_val

    if sample_wise:
        cols_scaled = []
        for col in cols:
            col_min = col.min()
            col_max = col.max()

            # Get scale, avoid zero division
            scale = col_max - col_min
            if scale == 0:
                # scale = 1
                cols_scaled.append(np.full_like(col, min_val))
            else:
                cols_scaled.append((col - col_min) / scale * val_scale + min_val)

    else:
        cols_concat = np.concatenate(cols, axis=0)
        cols_concat_min = cols_concat.min()
        cols_concat_max = cols_concat.max()

        # Get scale, avoid zero division
        scale = cols_concat_max - cols_concat_min
        if scale == 0:
            # scale = 1
            cols_scaled = [np.full_like(col, min_val) for col in cols]
        else:
            cols_scaled = [(col - cols_concat_min) / scale * val_scale + min_val for col in cols]

    return cols_scaled


def _save_to_fcs(
        fcs_dfs: List[pd.DataFrame],
        val_range: Tuple[float, float],
        save_path: str,
        save_filenames: Union[str, List[str]],
        fcs_metadata_dicts: Union[Dict, List[Dict]],
        sample_wise: bool,
):

    if sample_wise:
        for fcs_df, save_filename, fcs_metadata_dict in zip(fcs_dfs, save_filenames, fcs_metadata_dicts):
            _df_to_fcs(
                df=fcs_df,
                val_range=val_range,
                save_path=save_path,
                save_filename=save_filename,
                fcs_metadata_dict=fcs_metadata_dict,
            )
    else:

        fcs_df_concat = pd.concat(fcs_dfs, axis=0, ignore_index=True)

        _df_to_fcs(
            df=fcs_df_concat,
            val_range=val_range,
            save_path=save_path,
            save_filename=save_filenames,
            fcs_metadata_dict=fcs_metadata_dicts,
        )


def _df_to_fcs(
        df: pd.DataFrame,
        val_range: Tuple[float, float],
        save_path: str,
        save_filename: str,
        fcs_metadata_dict: Union[Dict, List[Dict]],
):

        # Add the correct range to the metadata
        fcs_metadata_dict.update({f"P{i}R": str(val_range[1]) for i in range(1, df.shape[1] + 1)})

        with open(os.path.join(save_path, save_filename), 'wb') as f:
            flowio.create_fcs(
                file_handle=f,
                event_data=df.to_numpy().flatten().tolist(),
                channel_names=df.columns.tolist(),
                opt_channel_names=df.columns.tolist(),
                metadata_dict=fcs_metadata_dict,
            )

