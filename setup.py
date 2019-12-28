from setuptools import setup

def readlines(file):
    with open(file) as f:
        return f.readlines()

setup (
    name = 'latticemc',
    version = '0.1',
    author = 'Karol Trojanowski',
    author_email = 'trojanowski.ifuj@gmail.com',
    license = 'MIT',
    keywords = 'lattice monte carlo biaxial nematic chirality',
    packages = ['latticemc'],
    install_requires = readlines('requirements.txt')
)