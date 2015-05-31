from distutils.core import setup
setup(
  name = 'grmpy',
  packages = ['grmpy', 'grmpy.tests', 'grmpy.tools', 'grmpy.user',], # this must be the same as the name above
  package_data = {'data': ['test.*'],},
  version = 'v1.1.1.4',
  description = 'Estimator for the generalized Roy Model',
  author = 'Philipp Eisenhauer',
  author_email = 'eisenhauer@policy-lab.org',
  url = 'https://github.com/grmToolbox/package', # use the URL to the github repo
  download_url = 'https://github.com/grmToolbox/package/tarball/v1.1.1', # I'll explain this in a second
  keywords = ['Generalized Roy', 'Policy Evaluation', 'Economics'], # arbitrary keywords
  classifiers = [],
  install_requires=['coveralls','numdifftools','numpy','scipy','statsmodels','nose','pandas','matplotlib',],
)
