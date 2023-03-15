from setuptools import setup

setup(
    name="psanalysis",
    version="0.0",
    description="Tools for analyzing planar scintigraphy scans",
    author="Monica Shapiro",
    author_email="mos57@pitt.edu",
    packages=["psanalysis"],
    install_requires=[
        "numpy >= 1.22",
        "scipy >= 1.8"
    ],
    tests_require=["pytests"]
)
