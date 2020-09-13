from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read()

setup(
    name='traffic',
    version='0.0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/jimparr19/traffic',
    license='',
    author='Jim Parr',
    author_email='jimparr19@gmail.com',
    description='Python package for simulating traffic',
    install_requires=requirements,
    package_data={
        'traffic': ['data/*.csv'],
    },
    python_requires=">=3.7.*",
)
