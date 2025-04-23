
import numpy as np
import scanpy as sc

from .._legacy_typing import List, Tuple, Union


def export_to_fcs(
        data_list: List[sc.AnnData],
        y_preds: Union[List[np.ndarray], None] = None,
        dim_red_coords: Union[List[List[np.ndarray]], None] = None,  # optional, list of lists of np arrays (multiple dim reds possible)
        dim_red_names: Union[List[str], None] = None,  # optional, list of names
        other_annotations: Union[List[Union[int, float, str]], None] = None,
        sample_wise: bool = False,  # whether to save to individual or one fcs file
        # sample_name_id: List[Tuple[int, str]],  # optional, tuple (sample_id, fn) -> also return df, do this on the fly
):

    # Todo

    return







