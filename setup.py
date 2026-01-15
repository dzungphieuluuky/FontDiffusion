from setuptools import setup, find_packages

setup(
    name="fontdiffusion",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "diffusers>=0.20.0",
        "transformers>=4.30.0",
        "accelerate>=0.21.0",
        "pillow>=10.0.0",
        "gradio>=4.0.0",
    ],
    entry_points={
        'console_scripts': [
            'fontdiffuser=fontdiffusion.cli:main',
        ],
    },
)