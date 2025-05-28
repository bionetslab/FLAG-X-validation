
import os
import flowio
import numpy as np
import pandas as pd
import scanpy as sc

from .._legacy_typing import List, Tuple, Dict, Union


def export_to_fcs(
        data_list: List[sc.AnnData],
        layer_key: Union[str, None] = None,
        val_range: Tuple[float, float] = (0.0, 2**20),
        keep_unscaled: bool = False,
        sample_wise: bool = False,  # whether to save to individual or one fcs file
        y_preds: Union[List[np.ndarray], None] = None,
        dim_red_coords: Union[List[List[np.ndarray]], None] = None,  # optional, list of lists of np arrays (multiple dim reds possible)
        dim_red_names: Union[List[str], None] = None,
        other_annotations: Union[List[List[np.ndarray]], None] = None,
        other_annotations_names: Union[List[str], None] = None,
        save_path: Union[str, None] = None,
        save_filenames: Union[str, List[str], None] = None,
        fcs_metadata_dicts: Union[Dict, List[Dict], None] = None,
) -> Union[List[pd.DataFrame], Tuple[List[pd.DataFrame], pd.DataFrame]]:

    if y_preds is None and dim_red_coords is None  and other_annotations is None:
        raise ValueError("Either 'y_preds' or 'dim_red_coords' or 'other_annotations' must not be None.")

    fcs_dfs = _init_fcs_dfs(data_list=data_list, layer_key=layer_key)

    annotations = []

    # Add sample ids
    if not sample_wise:
        sample_ids = []
        fns = []
        for i, adata in enumerate(data_list):
            fn = adata.uns['filename']
            fns.append(fn)
            sample_ids.append(np.full(adata.shape[0], i + 1))

        annotations.append((sample_ids, 'sample_id'))

        sample_fn_id_df = pd.DataFrame({'filenames': fns, 'sample_id': range(1, len(fns) + 1)})

    if y_preds is not None:
        annotations.append((y_preds, 'pred'))

    if dim_red_coords is not None:
        if dim_red_names is not None:
            if len(dim_red_coords) != len(dim_red_names):
                raise ValueError("Mismatch: 'dim_red_coords' and 'dim_red_names' must have the same length.")
        else:
            dim_red_names = [f'dim_red_{i}' for i in range(len(dim_red_coords))]

        for dim_red_coord, dim_red_name in zip(dim_red_coords, dim_red_names):

            dim_red_coord_0 = [x[:, 0] for x in dim_red_coord]
            dim_red_coord_1 = [x[:, 1] for x in dim_red_coord]

            annotations.append((dim_red_coord_0, dim_red_name + '_1'))
            annotations.append((dim_red_coord_1, dim_red_name + '_2'))

    if other_annotations is not None:
        if other_annotations_names is not None:
            if len(other_annotations) != len(other_annotations_names):
                raise ValueError("Mismatch: 'other_annotations' and 'other_annotations_names' must have the same length.")
        else:
            other_annotations_names = [f'annotation_{i}' for i in range(len(other_annotations))]

        for anno, anno_name in zip(other_annotations, other_annotations_names):
            annotations.append((anno, anno_name))

    for anno, anno_name in annotations:
        _add_columns(
            fcs_dfs=fcs_dfs,
            cols=anno,
            col_name=anno_name,
            val_range=val_range,
            keep_unscaled=keep_unscaled,
            sample_wise=sample_wise,
        )

    if save_path is None:
        save_path = os.getcwd()

    if save_filenames is None:
        if sample_wise:
            save_filenames = ['annotated_' + adata.uns['filename'] for adata in data_list]
        else:
            save_filenames = 'annotated.fcs'

    if fcs_metadata_dicts is None:
        if sample_wise:
            fcs_metadata_dicts = [dict(), ] * len(data_list)
        else:
            fcs_metadata_dicts = dict()

    _save_to_fcs(
        fcs_dfs=fcs_dfs,
        val_range=val_range,
        save_path=save_path,
        save_filenames=save_filenames,
        fcs_metadata_dicts=fcs_metadata_dicts,
        sample_wise=sample_wise,
    )

    if not sample_wise:
        out = fcs_dfs, sample_fn_id_df
    else:
        out = fcs_dfs

    return out


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


def _add_columns(
        fcs_dfs: List[pd.DataFrame],
        cols: List[np.ndarray],
        col_name: str,
        val_range: Tuple[float, float],
        keep_unscaled: bool,
        sample_wise: bool,
) -> List[pd.DataFrame]:

    # Scale the columns to the desired range
    scaled_cols = _scale_columns(cols=cols, val_range=val_range, sample_wise=sample_wise)

    for fcs_df, scaled_col, col in zip(fcs_dfs, scaled_cols, cols):

        fcs_df[col_name] = scaled_col

        if keep_unscaled:
            fcs_df[col_name + '_unscaled'] = col

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

