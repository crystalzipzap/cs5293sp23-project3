from setuptools import setup, find_packages

setup(
	name='project3',
	version='1.0',
	author='Chenyi Crystal Zhang',
	authour_email='cschmidt@ou.edu',
	packages=find_packages(exclude=('tests', 'docs')),
	setup_requires=['pytest-runner'],
	tests_require=['pytest']	
)