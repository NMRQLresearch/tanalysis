from setuptools import setup

setup(name='tanalysis',
      version='1.0',
      description='Data analysis and transformation via tensor network decompositions',
      author='Ryan Sweke',
      author_email='rsweke@gmail.com',
      url='https://github.com/NMRQLresearch/tanalysis',
      license='MIT',
      packages=['tanalysis'],
      install_requires=[
        'numpy',
        'sympy',
        'tncontract'],
      zip_safe=False)
