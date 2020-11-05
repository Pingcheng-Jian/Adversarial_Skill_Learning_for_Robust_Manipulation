from setuptools import find_packages, setup

# Required dependencies
required = [
    # Please keep alphabetized
    'torch',
    'gym>=0.15.4',
    'mujoco-py',
    'numpy',
    'tensorboardX',
    'mpi4py'
]

# Development dependencies
extras = dict()
extras['dev'] = [
    # Please keep alphabetized
    'ipdb',
    'memory_profiler',
    'pylint',
    'pyquaternion==0.9.5',
    'pytest>=4.4.0',  # Required for pytest-xdist
    'pytest-xdist',
]

setup(
    name='collabot',
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
    extras_require=extras,
)
