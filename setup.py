import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RollingHorizonEA",
    version="0.1.2",
    author="Chris Bamford",
    author_email="chrisbam4d@gmail.com",
    description="Rolling Horizon Evolutionary Algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bam4d/RollingHorizonEvolutionaryAlgorithm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)