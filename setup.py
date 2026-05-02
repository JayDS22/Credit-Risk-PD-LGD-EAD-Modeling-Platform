from setuptools import setup, find_packages

setup(
    name='credit-risk-pd-lgd-ead-platform',
    version='1.0.0',
    author='Jay Guwalani',
    description='Basel III Credit Risk PD/LGD/EAD Modeling Platform',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'scipy>=1.10.0',
        'scikit-learn>=1.3.0',
        'xgboost>=1.7.0',
        'statsmodels>=0.14.0',
        'lifelines>=0.27.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'plotly>=5.15.0',
        'pyyaml>=6.0',
        'joblib>=1.3.0',
        'tqdm>=4.65.0',
    ],
    extras_require={
        'test': ['pytest>=7.4.0', 'pytest-cov>=4.1.0'],
    },
)
