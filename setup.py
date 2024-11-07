from setuptools import setup, find_packages
import os

# Read version from __init__.py
with open(os.path.join('debugai', '__init__.py'), 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('"\'')
            break

# Read README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='debugai',
    version=version,
    author='Sami Halawa',
    author_email='author@example.com',
    description='AI-powered Python code analysis tool',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/debugai',
    packages=find_packages(exclude=['tests*']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.6',
    install_requires=[
        'click>=7.0',
        'ast-decompiler>=0.4.0',
        'pyyaml>=5.1',
        'chardet>=3.0',
        'psutil>=5.8',
        'ratelimit>=2.2'
    ],
    entry_points={
        'console_scripts': [
            'debugai=debugai.cli.commands:cli',
        ],
    },
    include_package_data=True,
    package_data={
        'debugai': ['.debugai.yml'],
    },
) 