#!/usr/bin/env python3
import os
import inspect
import importlib

from docutils.core import publish_parts


def walkdir(path, **kwargs):
    """ Walk directory

    Args:
        path (str): Path to walk
        ignore (List of str): List of files to ignore
        ext (str)[optional]: Filter file extensions

    Returns:

        List of files

    """
    ignore = kwargs.get("ignore", [])
    ext = kwargs.get("ext", ".py")

    files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if filename in ignore:
                pass
            elif ext is not None and filename.endswith(ext):
                files.append(os.sep.join([dirpath, filename]))
            elif ext is None:
                files.append(os.sep.join([dirpath, filename]))

    files.reverse()
    return files

script_path = os.path.dirname(os.path.realpath(__file__))
# print(walkdir(script_path + "/prototype")[0])
f = walkdir(script_path + "/prototype")[0]

print(f)
module = importlib.import_module("prototype.data.dataset")

functions = []
classes = []
for name, obj in inspect.getmembers(module):
    if inspect.isfunction(obj):
        functions.append(obj)
    if inspect.isclass(obj):
        classes.append(obj)

for func in functions:
    docstring = inspect.cleandoc(func.__doc__)
    html = publish_parts(docstring, writer_name='html')['html_body']
    print(html)
