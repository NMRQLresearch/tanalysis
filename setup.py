from setuptools import setup

setup(name='tanalysis',
      version='0.1',
      description='Data Compression with MPS core extraction',
      author='Ryan Sweke',
      author_email='rsweke@gmail.com',
      license='MIT',
      packages=['tanalysis'],
      install_requires=[
        'numpy',
        'sympy',
        'tncontract'],
      zip_safe=False)
