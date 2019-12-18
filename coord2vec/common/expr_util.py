from coord2vec.common import file_util
from coord2vec.common.file_util import list_all_files_in_folder
from coord2vec.config import TMP_EXPR_FILES_DIR, PROJECT_ROOT


def save_all_py_files():
    file_util.makedirs(TMP_EXPR_FILES_DIR, exists_ok=True)
    python_files_in_dir = list_all_files_in_folder(PROJECT_ROOT, "py", recursively=True)
    file_util.copy_files_while_keeping_structure(python_files_in_dir, PROJECT_ROOT, TMP_EXPR_FILES_DIR)
