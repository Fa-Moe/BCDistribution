from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Discrete Generalized Beta Cocho Distribution Modeling and Plotting'
LONG_DESCRIPTION = 'Process abundance data and estimate the parameters of a discrete generalized beta distribution (DGBD) (Martinez-Mekler et al., 2009) <doi:10.1371/journal.pone.0004791> that fits the data. Generates linear or non-linear model reports and uses matplotlib to plot rank abundance diagrams.'

# Setting up
setup(
        license_files= ('',),
        name="BCDistribution",
        version=VERSION,
        author="Francisco Farell Moedano Vargas",
        author_email="<farell.mova@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['pandas','numpy','statmodels','matplotlib','copy','os','json'], # add any additional packages that
        # needs to be installed along with your package. Eg: 'caer'
        keywords=['python', 'Beta Distribution', 'Rank-Abundance Diagram', ''],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: Microsoft :: Windows",
        ]
)
