from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os
import numpy
import sysconfig

PATH_TO_PKG = ''
SOURCES = ("vdbosch04_pair_finder_engine.pyx", "analytic_model.pyx")
THIS_PKG_NAME = ''


def get_extensions():

    names = [src.replace('.pyx', '') for src in SOURCES]
    sources = [os.path.join(PATH_TO_PKG, srcfn) for srcfn in SOURCES]
    include_dirs = ['numpy', numpy.get_include()]
    libraries = ['gsl', 'gslcblas']
    language = 'c++'
    extra_compile_args = ['-Ofast', '-march=native']

    extensions = []
    for name, source in zip(names, sources):
        extensions.append(Extension(name=name,
                          sources=[source],
                          include_dirs=include_dirs,
                          libraries=libraries,
                          language=language,
                          extra_compile_args=extra_compile_args))

    return extensions


def get_ext_filename_without_platform_suffix(filename):
    name, ext = os.path.splitext(filename)
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')

    if ext_suffix == ext:
        return filename

    ext_suffix = ext_suffix.replace(ext, '')
    idx = name.find(ext_suffix)

    if idx == -1:
        return filename
    else:
        return name[:idx] + ext


class BuildExtWithoutPlatformSuffix(build_ext):
    def get_ext_filename(self, ext_name):
        filename = super().get_ext_filename(ext_name)
        return get_ext_filename_without_platform_suffix(filename)

setup(
    ext_modules=get_extensions(),
    cmdclass={'build_ext': BuildExtWithoutPlatformSuffix},
)
