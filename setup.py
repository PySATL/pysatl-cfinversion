from setuptools import setup, find_packages

import json
import os


def read_pipenv_dependencies(fname):
    """Получаем из Pipfile.lock зависимости по умолчанию."""
    filepath = os.path.join(os.path.dirname(__file__), fname)
    with open(filepath) as lockfile:
        lockjson = json.load(lockfile)
        return [dependency for dependency in lockjson.get('default')]


if __name__ == '__main__':
    setup(
        name='CFInvert',
        version=os.getenv('PACKAGE_VERSION', '0.0.dev0'),
        package_dir={'': 'CFInvert'},
        packages=find_packages('CFInvert', include=[
            'CFInvert*'
        ]),
        description='A package for characteristic functions inversion.',
        install_requires=[
            *read_pipenv_dependencies('Pipfile.lock'),
        ]
    )
