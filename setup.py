from setuptools import setup, find_packages

setup(
    name='rao_algorithms',
    version='0.1.0',
    author='Sandeep Kunkunuru',
    author_email='sandeep.kunkunuru@gmail.com',
    description='BMR and BWR optimization algorithms with constraint handling',
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
)
