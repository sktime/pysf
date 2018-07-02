import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pysf",
    version="0.0.3",
    author="Ahmed Guecioueur",
    author_email="ucakaag@ucl.ac.uk",
    description="Supervised Forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ahmedgc/pysf",
    packages=setuptools.find_packages(),
	license="MIT",
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)