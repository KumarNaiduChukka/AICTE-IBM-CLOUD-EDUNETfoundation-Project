import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_union
from autoai_libs.transformers.exportable import ColumnSelector
from autoai_libs.transformers.exportable import NumpyColumnSelector
from autoai_libs.transformers.exportable import CompressStrings
from autoai_libs.transformers.exportable import NumpyReplaceMissingValues
from autoai_libs.transformers.exportable import NumpyReplaceUnknownValues
from autoai_libs.transformers.exportable import boolean2float
from autoai_libs.transformers.exportable import CatImputer
from autoai_libs.transformers.exportable import CatEncoder
from autoai_libs.transformers.exportable import float32_transform
from autoai_libs.transformers.exportable import FloatStr2Float
from autoai_libs.transformers.exportable import NumImputer
from autoai_libs.transformers.exportable import OptStandardScaler
from autoai_libs.transformers.exportable import NumpyPermuteArray
from snapml import SnapRandomForestClassifier

def get_pipeline():
    """
    This function returns the scikit-learn pipeline for the model.
    The pipeline is defined as in the notebook.

    :return: The scikit-learn pipeline for the model.
    :rtype: sklearn.pipeline.Pipeline
    """
    # Set CPU_NUMBER to a reasonable default
    CPU_NUMBER = 4

    # Define the pipeline steps as in the notebook
    column_selector_0 = ColumnSelector(
        columns_indices_list=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21,
            22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
            39, 40,
        ]
    )
    numpy_column_selector_0 = NumpyColumnSelector(
        columns=[1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    )
    compress_strings = CompressStrings(
        compress_type="hash",
        dtypes_list=[
            "char_str", "char_str", "char_str", "float_int_num", "float_int_num",
            "float_int_num", "float_int_num", "float_int_num", "float_int_num",
            "float_int_num", "float_int_num", "float_int_num", "float_int_num",
            "float_int_num", "float_int_num", "float_int_num", "float_int_num",
        ],
        missing_values_reference_list=["", "-", "?", float("nan")],
        misslist_list=[
            [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
        ],
    )
    numpy_replace_missing_values_0 = NumpyReplaceMissingValues(
        filling_values=float("nan"), missing_values=[]
    )
    numpy_replace_unknown_values = NumpyReplaceUnknownValues(
        filling_values=float("nan"),
        filling_values_list=[
            float("nan"), float("nan"), float("nan"), 100001, 100001, 100001,
            100001, 100001, 100001, 100001, 100001, 100001, 100001, 100001,
            100001, 100001, 100001,
        ],
        missing_values_reference_list=["", "-", "?", float("nan")],
    )
    cat_imputer = CatImputer(
        missing_values=float("nan"),
        sklearn_version_family="1",
        strategy="most_frequent",
    )
    cat_encoder = CatEncoder(
        dtype=np.float64,
        handle_unknown="error",
        sklearn_version_family="1",
        encoding="ordinal",
        categories="auto",
    )
    pipeline_0 = make_pipeline(
        column_selector_0,
        numpy_column_selector_0,
        compress_strings,
        numpy_replace_missing_values_0,
        numpy_replace_unknown_values,
        boolean2float(),
        cat_imputer,
        cat_encoder,
        float32_transform(),
    )
    column_selector_1 = ColumnSelector(
        columns_indices_list=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21,
            22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
            39, 40,
        ]
    )
    numpy_column_selector_1 = NumpyColumnSelector(
        columns=[
            0, 4, 5, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
            35, 36, 37, 38,
        ]
    )
    float_str2_float = FloatStr2Float(
        dtypes_list=[
            "float_int_num", "float_int_num", "float_int_num", "float_int_num",
            "float_int_num", "float_num", "float_num", "float_num", "float_num",
            "float_num", "float_num", "float_num", "float_int_num",
            "float_int_num", "float_num", "float_num", "float_num", "float_num",
            "float_num", "float_num", "float_num", "float_num",
        ],
        missing_values_reference_list=[],
    )
    numpy_replace_missing_values_1 = NumpyReplaceMissingValues(
        filling_values=float("nan"), missing_values=[]
    )
    num_imputer = NumImputer(missing_values=float("nan"), strategy="median")
    opt_standard_scaler = OptStandardScaler(use_scaler_flag=False)
    pipeline_1 = make_pipeline(
        column_selector_1,
        numpy_column_selector_1,
        float_str2_float,
        numpy_replace_missing_values_1,
        num_imputer,
        opt_standard_scaler,
        float32_transform(),
    )
    union = make_union(pipeline_0, pipeline_1)
    numpy_permute_array = NumpyPermuteArray(
        axis=0,
        permutation_indices=[
            1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 4, 5,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
            37, 38,
        ],
    )
    snap_random_forest_classifier = SnapRandomForestClassifier(
        compress_trees=True,
        gpu_ids=np.array([0], dtype=np.uint32),
        max_depth=10,
        n_jobs=CPU_NUMBER,
        random_state=33,
    )

    pipeline = make_pipeline(
        union, numpy_permute_array, snap_random_forest_classifier
    )

    return pipeline
