from distutils.core import setup
setup(
  name = 'grmpy',
  packages = ['grmpy'], # this must be the same as the name above
  version = '0.1',
  description = 'Estimator for the generalized Roy Model',
  author = 'Philipp Eisenhauer',
  author_email = 'eisenhauer@policy-lab.org',
  url = 'https://github.com/grmToolbox/package', # use the URL to the github repo
  download_url = 'https://github.com/grmToolbox/package/tarball/0.1', # I'll explain this in a second
  keywords = ['Generalized Roy', 'Policy Evaluation', 'Economics'], # arbitrary keywords
  classifiers = [],
  install_requires=['coveralls','numdifftools','numpy','scipy','statsmodels','nose','pandas','matplotlib',],
)
