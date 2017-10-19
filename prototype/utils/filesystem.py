import os


def walkdir(path, ext=None):
    """ Walk directory

    Args:

        path (str): Path to walk
        ext (str)[optional]: Filter file extensions

    Returns:

        List of files

    """
    files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if ext is not None and filename.endswith(ext):
                files.append(os.sep.join([dirpath, filename]))
            elif ext is None:
                files.append(os.sep.join([dirpath, filename]))

    files.reverse()
    return files
