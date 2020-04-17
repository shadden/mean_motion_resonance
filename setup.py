from distultils.core import setup

setup(
        name='MeanMotionResonance',
        version='0.1dev',
        packages=['MeanMotionResonance',],
        author='Sam Hadden', 
        author_email='samuel.hadden@cfa.harvard.edu',
        license='GPL',
        classifiers=[
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 3 - Alpha',
            # Indicate who your project is intended for
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'Topic :: Software Development :: Build Tools',
            'Topic :: Scientific/Engineering :: Astronomy',
            # Pick your license as you wish (should match "license" above)
            'License :: MIT',
            # Specify the Python versions you support here. In particular, ensure
            # that you indicate whether you support Python 2, Python 3 or both.
            'Programming Language :: Python :: 3',
        ],
        keywords='astronomy astrophysics',
        install_requires=['numpy','rebound','exoplanet','theano'],
        include_package_data=True,
        long_description=open('README.md').read(),
        zip_safe=False
    )
