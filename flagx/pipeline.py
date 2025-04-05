
import os
import warnings

import matplotlib.pyplot as plt

from typing import List, Union, Literal

from .io import FlowDataManager

class GatingPipeline:
    def __init__(
            self,
            gating_method: Literal['som', 'fcnn_softmax'] = 'som',
            gating_method_kwargs: Union[dict, None] = None,

    ):
        super().__init__()

        # Gating method
        self.gating_method = gating_method
        self.gating_method_kwargs = gating_method_kwargs
        if self.gating_method_kwargs is None:
            self.gating_method_kwargs = {}

        # ### Train attributes
        # Train data manager
        self.is_trained_ = False
        self.train_data_manager_ = None
        self.train_data_file_names_ = None
        self.train_data_file_type_ = None
        self.train_data_file_path_ = None
        self.train_data_manager_save_path_ = None



        # Gating method
        self.gater_ = None


        # Todo:
        #  - Load and process data
        #  - Train on processed train data
        #  - When new data is presented:
        #    - Gate
        #    - Compute dim red
        #    - Export to fcs (gating and 2d coordinates), add function to fdm!!!
        #  - For the special case where SOM should be used as template add function export SOM

    def data_pipeline(self):

        return

    def train(
            self,
            train_data_file_names: List[str],
            train_data_file_type: Union[Literal['fcs', 'csv'], None] = None,
            train_data_file_path: Union[str, None] = None,
            train_data_manager_save_path: Union[str, None] = None,
            channel_names_alignment_kwargs: Union[dict, None] = None,  # {'reference_channel_names': int or dict}
            relabel_data_kwargs: Union[dict, None] = None,  # {'label_key': str, 'new_label_key': str, 'old_to_new_label_mapping': dict}
            preprocessing_kwargs: Union[dict, None] = None,  # {'flavour': str, !optional! 'flavour_kwargs': dict, !optional! 'save_raw_to_layer': str}
    ):

        # Instantiate the train data manager
        self.train_data_file_names_ = train_data_file_names
        self.train_data_file_type_ = train_data_file_type
        self.train_data_file_path_ = train_data_file_path
        self.train_data_manager_save_path_ = train_data_manager_save_path

        self.train_data_manager_ = FlowDataManager(
            data_file_names=self.train_data_file_names_,
            data_file_type=self.train_data_file_type_,
            data_file_path=self.train_data_file_path_,
            save_path=self.train_data_manager_save_path_,
        )

        # Load train data files to anndata
        self.train_data_manager_.load_data_files_to_anndata()

        # Check the number of events per sample
        self.train_data_manager_.check_sample_sizes(filename_sample_sizes_df='train_sample_sizes.csv')
        self.train_data_manager_.plot_sample_size_df(sample_size_df=self.train_data_manager_.sample_sizes_, dpi=300)
        plt.tight_layout()
        plt.savefig(os.path.join(self.train_data_manager_.save_path, 'train_sample_sizes.png'))

        # Align channel name
        channel_names_alignment_kwargs = {} if channel_names_alignment_kwargs is None else channel_names_alignment_kwargs
        reference_channel_names = channel_names_alignment_kwargs.get('reference_channel_names', None)
        self.train_data_manager_.align_channel_names(
            reference_channel_names=reference_channel_names,  # None -> use 1st entry of train data list as reference
            filename_log_df='train_og_channel_names.csv',
        )

        # Relabel data if relabel_data_kwargs is not None
        if relabel_data_kwargs is not None:

            old_to_new_label_mapping = relabel_data_kwargs['old_to_new_label_mapping']
            label_key = relabel_data_kwargs['label_key']
            new_label_key = relabel_data_kwargs['new_label_key']

            self.train_data_manager_.relabel_data(
                data_set='all',
                old_to_new_label_mapping=old_to_new_label_mapping,
                label_key=label_key,
                label_layer_key=None,  # No preprocessing done yet
                new_label_key=new_label_key,
            )
            current_lk = new_label_key
        else:
            current_lk = label_key


        # Apply sample wise preprocessing transformation, if preprocessing_kwargs is not None
        if preprocessing_kwargs is not None:
            flavour = preprocessing_kwargs['flavour']
            save_raw_to_layer = preprocessing_kwargs.get('save_raw_to_layer', 'raw')
            flavour_kwargs = preprocessing_kwargs.get('flavour_kwargs', {})

            self.train_data_manager_.sample_wise_preprocessing(
                flavour=flavour,
                save_raw_to_layer=save_raw_to_layer,
                **flavour_kwargs,
            )







    def gate(self):
        pass


    def reduce_dimension(self):
        pass



