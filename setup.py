import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "sasquatch",
    version = "0.0.1",
    author = "Kevin Ellis",
    author_email = "ellisk@mit.edu",
    description = ("unsupervised program synthesis using Z3"),
    license = "",
    keywords = "Z3 synthesis learning",
    url = "http://github.com/ellisk42/sasquatch",
    packages=['sasquatch'],
    long_description=read('README'),
    # see the list at: https://pypi.python.org/pypi?:action=list_classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Natural Language :: English"
    ],
)
