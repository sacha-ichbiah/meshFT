import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from setuptools import setup,find_packages
from pathlib import Path
this_directory = Path(__file__).parent
readme = (this_directory / "README.md").read_text()

if not torch.cuda.is_available():
    setup(
        name='MeshFT',
        version='0.1.1',
        description='MeshFT implementation',
        url='https://github.com/sacha-ichbiah/meshFT',
        author='Sacha Ichbiah',
        author_email='sacha.ichbiah@polytechnique.org',
        license='BSD',
        include_package_data=True,
        packages = find_packages(),

        install_requires=['numpy>=1.21.6',
                    'torch>=1.13.1',
                        ],
        long_description=readme,
        long_description_content_type="text/markdown"
    )

else : 
    setup(
        name='MeshFT',
        version='0.1.1',
        description='MeshFT implementation',
        url='https://github.com/sacha-ichbiah/meshFT',
        author='Sacha Ichbiah',
        author_email='sacha.ichbiah@polytechnique.org',
        license='BSD',
        include_package_data=True,
        packages = find_packages(),

        ext_modules=[
            CUDAExtension(
                'meshft.fourier3d_cpp',
                sources = ['meshft/cpp/fourier3d_cuda.cpp','meshft/cpp/fourier3d_cuda_kernel.cu'],
                extra_compile_args={'cxx': ['-g'],
                                    'nvcc': ['-O2']}),
        ],
        cmdclass={'build_ext': BuildExtension,
        },

        install_requires=['numpy>=1.21.6',
                    'torch>=1.13.1',
                        ],
        long_description=readme,
        long_description_content_type="text/markdown"
    )