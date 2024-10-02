from setuptools import setup, find_packages
import os

# Read the contents of README.md for PyPI project description
def read_file(filename):
    with open(filename, "r", encoding="utf-8") as fh:
        return fh.read()

setup(
    name='rao_algorithms',
    version='0.2.2',
    author='Sandeep Kunkunuru',
    author_email='sandeep.kunkunuru@gmail.com',
    description='BMR and BWR optimization algorithms with constraint handling',
    long_description=read_file("README.md"),  # Load README.md as the long description
    long_description_content_type='text/markdown',  # Specify the format of the long description
    url='https://github.com/VaidhyaMegha/optimization_algorithms',  # URL to GitHub repo
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    project_urls={  # Additional links to display on PyPI page
        'Documentation': 'https://github.com/VaidhyaMegha/optimization_algorithms/wiki',
        'Source': 'https://github.com/VaidhyaMegha/optimization_algorithms',
        'Bug Reports': 'https://github.com/VaidhyaMegha/optimization_algorithms/issues',
    },
)
