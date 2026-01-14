from .interleave_datasets import UnifiedEditIterableDataset, GroupPhotoUnifiedEditIterableDataset, SepPhotoTokenIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,  
    'data_specialtoken': SepPhotoTokenIterableDataset,
    'data_notoken': SepPhotoTokenIterableDataset,
}

DATASET_INFO = {
    'data_specialtoken':{
        'seedxedit_multi': {
            'data_dir': './data/data_specialtoken/seedxedit_multi',
            'num_files': 2177,
            'num_total_samples': 653237,
            "parquet_info_path": './data/data_specialtoken/parquet_info/parquet_info.json', # information of the parquet files
		},
    },
    'data_notoken':{
        'seedxedit_multi': {
            'data_dir': './data/data_notoken/seedxedit_multi',
            'num_files': 2177,
            'num_total_samples': 653237,
            "parquet_info_path": './data/data_notoken/parquet_info/parquet_info.json', # information of the parquet files
		},
    }
}