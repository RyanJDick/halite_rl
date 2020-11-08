import setuptools

setuptools.setup(
    name="halite-rl",
    version="0.0.1",
    author="Ryan Dick",
    author_email="ryanjdick3@gmail.com",
    description="Package for training RL agents to play the game 'Halite'.",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
)