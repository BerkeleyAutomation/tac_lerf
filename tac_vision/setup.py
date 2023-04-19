from setuptools import setup

setup(name='tacvis',
      version='0.1.0',
      description='Repo for tactile visual representation learning',
      package_dir={'': '.'},
      packages=['tacvis'],
      install_requires=[
        #   'torch==1.11.0',
          'pytorch-lightning',
          'torchvision',
          'Pillow',
          'wandb',
          'matplotlib',
          'opencv-python',
          'scikit-image',
          'einops'
      ]
      )
