import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = ["numpy", "pandas", "scikit-learn",
                "scipy", "joblib", "pyyaml",
                "multiprocess"]

setuptools.setup(
    name="mabfs",
    version="0.0.4",
    author="Evgeny Orlov",
    author_email="evgenyorlov1991@gmail.com",
    description="Multi armed bandit feature selection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/feodal01/Multi-armed-bandit-feature-selection",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
