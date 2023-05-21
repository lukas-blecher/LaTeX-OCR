#!/usr/bin/env python

import setuptools

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text(encoding='utf-8')

gui = [
    'PyQt6',
    'PyQt6-WebEngine',
    'pyside6',
    'pynput',
    'screeninfo',
]
api = [
    'streamlit>=1.8.1',
    'fastapi>=0.75.2',
    'uvicorn[standard]',
    'python-multipart'
]
train = [
    'python-Levenshtein>=0.12.2',
    'torchtext>=0.6.0',
    'imagesize>=1.2.0',
]
highlight = ['pygments']

setuptools.setup(
    name='pix2tex',
    version='0.1.2',
    description='pix2tex: Using a ViT to convert images of equations into LaTeX code.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Lukas Blecher',
    author_email='luk.blecher@gmail.com',
    url='https://github.com/lukas-blecher/LaTeX-OCR/',
    license='MIT',
    keywords=[
        'artificial intelligence',
        'deep learning',
        'image to text'
    ],
    packages=setuptools.find_packages(),
    package_data={
        'pix2tex': [
            'resources/*',
            'model/settings/*.yaml',
            'model/dataset/*.json',
        ]
    },
    install_requires=[
        'tqdm>=4.47.0',
        'munch>=2.5.0',
        'torch>=1.7.1',
        'opencv_python_headless>=4.1.1.26',
        'requests>=2.22.0',
        'einops>=0.3.0',
        'x_transformers==0.15.0',
        'transformers>=4.18.0',
        'tokenizers>=0.13.0',
        'numpy>=1.19.5',
        'Pillow>=9.1.0',
        'PyYAML>=5.4.1',
        'pandas>=1.0.0',
        'timm==0.5.4',
        'albumentations>=0.5.2',
        'pyreadline3>=3.4.1; platform_system=="Windows"',
    ],
    extras_require={
        'all': gui+api+train+highlight,
        'gui': gui,
        'api': api,
        'train': train,
        'highlight': highlight,
    },
    entry_points={
        'console_scripts': [
            'pix2tex_gui = pix2tex.__main__:main',
            'pix2tex_cli = pix2tex.__main__:main',
            'latexocr = pix2tex.__main__:main',
            'pix2tex = pix2tex.__main__:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
    ],
)
