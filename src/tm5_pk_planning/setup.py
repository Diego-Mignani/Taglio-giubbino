from setuptools import setup, find_packages

package_name = 'tm5_pk_planning'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='diego',
    maintainer_email='S1093616@studenti.univpm.it',
    description='Planning module for TM5',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'planning_node = tm5_pk_planning.nodes.planning_node:main',
        ],
    },
)
