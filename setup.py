from setuptools import setup, find_packages
import os

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = f"{lib_folder}/requirements.txt"
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

setup(
    name='pythia_tool',
    version='1.0.0',
    author="Cameron Cagan",
    author_email="",
    description="An LLM driven prompt improvement tool for optimizing LLM applications.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/clai-group/Pythia",  
    license="MIT",
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    keywords="llm prompt improvement ai iterative optimization",
    include_package_data=False,
    zip_safe=False,
    project_urls={
        "Documentation": "https://github.com/clai-group/Pythia/blob/main/README.md",
        "Source": "https://github.com/clai-group/Pythia",
        "Tracker": "https://github.com/clai-group/Pythia/issues",
    },
    extras_require={
        "ollama": ["ollama"],  # Adjust based on actual optional deps
        "google": ["google-auth", "google-genai"],
    },
)