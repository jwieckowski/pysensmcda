import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pysensmcda",
    version="1.0.0",
    author="Jakub Więckowski, Bartosz Paradowski",
    author_email="j.wieckowski@il-pib.pl, b.paradowski@il-pib.pl",
    description="Python Sensitivity Analysis Multi-Criteria Decision Analysis Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jwieckowski/pysensmcda",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        "joblib",
        "matplotlib",
        "npy_append_array",
        "numpy",
        "pandas",
        "pymcdm",
        "scipy",
        "setuptools",
        "thread6",
        "seaborn",
        "tqdm",
        "setuptools",
    ]
)
