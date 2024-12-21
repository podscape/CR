from setuptools import setup, find_packages

setup(
    name="news_aggregator",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'requests',
        'beautifulsoup4',
        'numpy',
        'feedparser',
        'fake-useragent',
    ],
)