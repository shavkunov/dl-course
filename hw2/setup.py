from distutils.core import setup

setup(
    name='resnext',
    version='0.1',
    packages=['resnext'],
    install_requires=[
        "pytorch >= 0.4.1",
        "torchvision",
        "tensorboardX"
    ],
    author='Mikhail Shavkunov',
	author_email='mv.shavkunov@yandex.ru'
)