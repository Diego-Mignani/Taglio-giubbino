from setuptools import find_packages, setup

package_name = 'tm5_pk_traiettorie'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'genera_traiettorie = tm5_pk_traiettorie.tm5_trajectory:main',
            'plot_trajectory = tm5_pk_traiettorie.tm5_plotter:main',
        ],

    },
)



