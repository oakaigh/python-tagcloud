import setuptools

setuptools.setup(
    python_requires='>=3.8',
    name='tagcloud',
    version='2.0.0',
    install_requires=[
        'numpy>=1.23.5',
        'numba>=0.56.4'
    ],
    extras_require={
        'matplotlib': ['matplotlib>=3.6.2'],
        'pillow': [
            'pillow>=9.4.0',
            #'freetype-py>=2.3.0'
        ]
    }
)
