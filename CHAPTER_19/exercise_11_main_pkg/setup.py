#! /usr/bin/env python3

from setuptools import find_packages, setup

REQUIRED_PACKAGES=[
    "numpy<=1.26.4",
    "tensorboard==2.16.2",
    'google-api-core==2.24.2',
    'google-auth==2.39.0',
    'google-cloud-aiplatform==1.89.0',
    'google-cloud-bigquery==3.31.0',
    'google-cloud-core==2.4.3',
    'google-cloud-resource-manager==1.14.2',
    'google-cloud-storage==2.19.0',
    'google-crc32c==1.7.1',
    'google-pasta==0.2.0',
    'google-resumable-media==2.7.2',
    'googleapis-common-protos==1.66.0',
    "gymnasium==1.1.1",
    "requests==2.32.3",
    "urllib3==2.3.0"
    ]

setup(
    name = "example_pkg_h",
    version= "0.1",
    packages=find_packages(), 
    include_package_data=True,
    description="Training application",
    install_requires=REQUIRED_PACKAGES
)

