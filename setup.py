from pathlib import Path
from typing import List

from setuptools import setup


def read_requirements(filename: str = "requirements.txt") -> List[str]:
    requirements_path = Path(__file__).with_name(filename)
    if not requirements_path.exists():
        return []
    # Ignore blank lines and comments to prevent setuptools errors.
    return [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


setup(
    name='tfbind',
    version='0.1',
    description='Transcription factor - dna binding analysis',
    # url='http://github.com/...',
    author='Ruben Solozabal',
    affiliation='MBZUAI',
    author_email='ruben.solozabal@mbzuai.ac.ae',
    license='MIT',
    packages=['src'],
    zip_safe=False,
    install_requires=read_requirements(),
)
