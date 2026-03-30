from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'tm5_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # 🔥 INSTALLA I FILE DI LAUNCH DALLA CARTELLA CORRETTA
        (os.path.join('share', package_name, 'launch'),
         glob('tm5_bringup/launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='diego',
    maintainer_email='S1093616@studenti.univpm.it',
    description='Bringup for TM5 simulation and real robot',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [],
    },
)
