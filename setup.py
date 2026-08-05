import pathlib
from distutils.util import convert_path

from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

plg_ns = {}
about_path = convert_path('beetsplug/essentia/about.py')
with open(about_path) as about_file:
    exec(about_file.read(), plg_ns)

# Setup
setup(
    name=plg_ns['__PACKAGE_NAME__'],
    version=plg_ns['__version__'],
    description=plg_ns['__PACKAGE_DESCRIPTION__'],
    author=plg_ns['__author__'],
    author_email=plg_ns['__email__'],
    url=plg_ns['__PACKAGE_URL__'],
    license='MIT',
    long_description=README,
    long_description_content_type='text/markdown',
    platforms='ALL',

    include_package_data=True,
    package_data={
        'beetsplug.essentia': ['config_default.yml'],
    },
    test_suite='test',
    packages=['beetsplug.essentia'],

    python_requires='>=3.12',

    install_requires=[
        'beets>=1.4.9',
        'pyyaml',
        'essentia-tensorflow',
    ],

    tests_require=[
        'pytest', 'nose', 'coverage',
        'mock', 'six', 'pyyaml',
    ],

    # Extras needed during testing
    extras_require={
        'tests': [],
    },

    classifiers=[
        'Topic :: Multimedia :: Sound/Audio',
        'License :: OSI Approved :: MIT License',
        'Environment :: Console',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
    ],
)
