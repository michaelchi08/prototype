#!/usr/bin/env python3
import os
import pydoc


def walk(path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith('.py') and filename != "__init__.py":
                files.append(os.sep.join([dirpath, filename]))
    files.reverse()
    return files


def function_docstr(fn):
    fn_name = fn[0]
    fn_args = pydoc.inspect.formatargspec(*pydoc.inspect.getargspec(fn[1]))
    fn_doc = fn[1].__doc__
    return {"type": "function",
            "name": fn_name,
            "args": fn_args,
            "docstr": fn_doc}


def method_docstr(mh):
    mh_name = mh[0]
    mh_args = pydoc.inspect.formatargspec(*pydoc.inspect.getargspec(mh[1]))
    mh_doc = mh[1].__doc__
    return {"type": "method",
            "name": mh_name,
            "args": mh_args,
            "docstr": mh_doc}


def class_docstr(cl):
    if cl[0] == "__class__" or cl[0].startswith("_"):
        return None

    cl_name = cl[0]
    cl_methods = []
    for mh in pydoc.inspect.getmembers(cl[1], pydoc.inspect.isfunction):
        cl_methods.append(method_docstr(mh))

    return {"type": "class",
            "name": cl_name,
            "methods": cl_methods}


def docstr(file_path):
    module_name = file_path.replace(".py", "").replace("/", ".")

    print("-> {}".format(module_name))
    # docstr = pydoc.render_doc(module_name, "%s", renderer=pydoc.plaintext)
    # docstr = "\n".join(docstr.split("\n")[5:-5])

    # load module
    module = pydoc.safeimport(module_name)
    if module is None:
        raise RuntimeError("Module {} not found!".format(module_name))

    # inspect class
    for cl in pydoc.inspect.getmembers(module, pydoc.inspect.isclass):
        docstr = class_docstr(cl)

    # for fn in pydoc.inspect.getmembers(module, pydoc.inspect.isfunction):
    #     docstr = function_docstr(fn)
    #     print(docstr)
    #     print()

    return (module_name, docstr)


# def genapi(module_name, docstr, output_dir="./"):
#     api_filename = module_name.replace(".", "_") + ".md"
#     api_doc = open(os.path.join(output_dir, api_filename), "w")
#     api_doc.write(docstr)
#     api_doc.close()


if __name__ == "__main__":
    files = walk("prototype")
    module_name, doc = docstr("prototype/vision/camera_models.py")
    # module_name, doc = docstr("prototype/vision/homography.py")
    # for f in files:
    #     module_name, doc = docstr(f)
        # genapi(module_name, doc, "./docs/api")
