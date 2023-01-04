import setuptools

setuptools.setup(
    python_requires='>=3.8',
    name='wordcloud',
    version='2.0.0',
    install_requires=[
        'numba>=0.56.4',
        'numpy>=1.23.5'
    ],
    extras_require={
        'matplotlib': ['matplotlib>=3.6.2'],
        'pillow': ['pillow>=9.4.0']
    }
)
