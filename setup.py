try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='finbot',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    description='A project for processing investment fund factsheets and answering questions about them',
    url='https://github.com/ese-msc-2023/irp-sl4623',
    author='Sara Lakatos',
    author_email='sl4623@imperial.ac.uk',
    packages=['finbot'],
    install_requires=[
        'python-dotenv',
        'pydantic==2.8.2',
        'langchain==0.2.10',
        'langchain-core==0.2.22',
        'langchain_community==0.2.9',
        'transformers==4.42.4',
        'qdrant-client',
        'langchain_qdrant',
        'langchain_huggingface',
        'pytest',
        'numpy',
        'scipy',
        'unstructured[all-docs]',
        'pdfplumber',
        'pandas',
        'PyPDF2',
        'tabula-py',
        'torch==2.3.1+cu121',
        'torchvision==0.18.1+cu121',
        'lxml',
        'langchainhub'
    ],
    extras_require={
        'dev': [
            'jupyter',
            'ipykernel',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
