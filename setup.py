import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
readme = (this_directory / "README.md").read_text()

common_kwargs = {
    'name': 'MeshFT',
    'version': '1.1.0',
    'description': 'MeshFT implementation',
    'url': 'https://github.com/sacha-ichbiah/meshFT',
    'author': 'Sacha Ichbiah',
    'author_email': 'sacha.ichbiah@polytechnique.org',
    'license': 'BSD',
    'include_package_data': True,
    'packages': find_packages(),
    'install_requires': ['numpy>=1.21.6', 'torch>=1.13.1'],
    'long_description': readme,
    'long_description_content_type': "text/markdown",
    'classifiers': [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
}

if not torch.cuda.is_available():
    setup(**common_kwargs)
else:
    setup(
        ext_modules=[
            CUDAExtension(
                'meshft.fourier3d_cpp',
                sources=['meshft/cpp/fourier3d_cuda.cpp', 'meshft/cpp/fourier3d_cuda_kernel.cu'],
                extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}),
        ],
        cmdclass={'build_ext': BuildExtension},
        **common_kwargs
    )

