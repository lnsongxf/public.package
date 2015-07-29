from distutils.core import setup
setup(
  name = 'grmpy',
  packages = ['grmpy', 'grmpy.tests', 'grmpy.tools', 'grmpy.tools.msc',
              'grmpy.tools.user', 'grmpy.tools.optimization'], # this
  # must be the same as the name above
  package_data = {'grmpy': ['data/test.*'],},
  version = '1.1.1.19',
  description = 'Estimator for the generalized Roy Model',
  author = 'Philipp Eisenhauer',
  author_email = 'eisenhauer@policy-lab.org',
  url = 'https://github.com/grmToolbox/package', # use the URL to the github repo
  keywords = ['Generalized Roy', 'Policy Evaluation', 'Economics'], # arbitrary keywords
  classifiers = [],
  install_requires=['matplotlib','numdifftools','numpy','scipy','statsmodels','nose','pandas', 'patsy'],
)
