import setuptools

requirements = [
    # Torch
    "torch==1.4",
    "torchvision==0.4",
    "pytorch-lightning",
    "test-tube",
    "torchsummary",
    # Data handling
    "numpy",
    "Pillow",
    "opencv-python",
    # Visualization
    "matplotlib",
    "seaborn",
    "pydot",
    "pydotplus",
    # Misc
    "IPython",
    "joblib",
    "natsort",
    "tqdm",
]

dev_requirements = [
    "jupyter",
    "black"
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kitti_adapt",
    version="0.0.1",
    author="Krishna Penukonda",
    url="",
    long_description=long_description,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={"dev": dev_requirements,},
)
