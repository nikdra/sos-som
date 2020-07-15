# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name="som",  # Replace with your own username
    version="20.06",
    author="Nikola Dragovic",
    author_email="e1528986@tuwien.ac.at",
    description="Implementation of a standard SOM algorithm with visualizations ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nikdra/sos-som",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.6, <4',
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/nikdra/sos-som/issues',
        'Source': 'https://github.com/nikdra/sos-som/',
    }
)
