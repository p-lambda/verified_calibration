import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="uncertainty-calibration",
    version="0.1.3",
    author="Ananya Kumar",
    author_email="skywalker94@gmail.com",
    description="Utilities to calibrate model uncertainties and measure calibration.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AnanyaKumar/verified_calibration",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'sklearn', 'parameterized'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
