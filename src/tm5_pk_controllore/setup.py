from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'tm5_pk_controllore'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/tm5_pk_controllore']),
        ('share/tm5_pk_controllore', ['package.xml']),
        ('share/tm5_pk_controllore/launch', glob('launch/*.py')),
        ('share/tm5_pk_controllore/config', glob('config/*.yaml')),
        ],

    
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='diego',
    maintainer_email='diego@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'comunica_robot = tm5_pk_controllore.tm5_comunicator:main',
            'controlla_robot = tm5_pk_controllore.tm5_controller:main',
        ],
    },
)
