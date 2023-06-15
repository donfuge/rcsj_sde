import setuptools

setuptools.setup(
    name="rcsj_sde",
    version="1.0",
    packages=["rcsj_sde"],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "numba",
        "tqdm",
        ],
)
