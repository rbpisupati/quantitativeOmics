"""
    Quantitative genentics toolkit
    ~~~~~~~~~~~~~
    A python toolkit for quantitative genetics
    Integrating various tools, limix, pygwas
    :copyright: Rahul Pisupati @ 2017
    :license:   GMI
"""
import os
import argparse
import sys
from pyquant.genopheno import lmm
from pyquant.genopheno import vca
import logging, logging.config

__version__ = '1.0.0'
__updated__ = "24.03.2017"
__date__ = "23.03.2017"

def setLog(logDebug):
  log = logging.getLogger()
  if logDebug:
    numeric_level = getattr(logging, "DEBUG", None)
  else:
    numeric_level = getattr(logging, "ERROR", None)
  log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
  lch = logging.StreamHandler()
  lch.setLevel(numeric_level)
  lch.setFormatter(log_format)
  log.setLevel(numeric_level)
  log.addHandler(lch)

def die(msg):
  sys.stderr.write('Error: ' + msg + '\n')
  sys.exit(1)

def get_options(program_license,program_version_message):
    inOptions = argparse.ArgumentParser(description=program_license)
    inOptions.add_argument('-V', '--version', action='version', version=program_version_message)
    subparsers = inOptions.add_subparsers(title='subcommands',description='Choose a command to run',help='Following commands are supported')

    vca_parser = subparsers.add_parser('vca_st', help="variance component analysis, single trait using limix")
    vca_parser.add_argument("-p", "--phenoFile", dest="phenoFile", help="File for phenotypes")
    vca_parser.add_argument("-g", "--genoFile", dest="genoFile", help="snp data")
    vca_parser.add_argument("-k", "--kinFile", dest="kinFile", help="kinship file based on snps", default = None)
    vca_parser.add_argument("-b", "--cisregion", dest="cisregion", help="cis region, Ex. Chr1,1,100")
    vca_parser.add_argument("-v", "--verbose", action="store_true", dest="logDebug", default=False, help="Show verbose debugging output")
    vca_parser.set_defaults(func=vca_singletrait)

    eqtl_parser = subparsers.add_parser('eqtl_st', help="Single trait eQTL mapping")
    eqtl_parser.add_argument("-p", "--phenoFile", dest="phenoFile", help="File for phenotypes")
    eqtl_parser.add_argument("-g", "--genoFile", dest="genoFile", help="snp data")
    eqtl_parser.add_argument("-k", "--kinFile", dest="kinFile", help="kinship file based on snps", default = None)
    eqtl_parser.add_argument("-b", "--cisregion", dest="cisregion", help="cis region, Ex. Chr1,1,100")
    eqtl_parser.add_argument("-v", "--verbose", action="store_true", dest="logDebug", default=False, help="Show verbose debugging output")
    eqtl_parser.set_defaults(func=eqtl_singletrait)

    lmm_parser = subparsers.add_parser('lmm_st', help="linear mixed model for single trait")
    lmm_parser.add_argument("-p", "--phenoFile", dest="phenoFile", help="File for phenotypes")
    lmm_parser.add_argument("-g", "--genoFile", dest="genoFile", help="snp data")
    lmm_parser.add_argument("-t", "--test", dest="test", help="test", default = "lrt")
    lmm_parser.add_argument("-k", "--kinFile", dest="kinFile", help="kinship file based on snps", default = None)
    lmm_parser.add_argument("-s", "--transformation", dest="transformation", help="transformation for the phenotypes", default = "most_normal")
    lmm_parser.add_argument("-o", "--outFile", dest="outFile", help="output file for pvalues")
    lmm_parser.add_argument("-v", "--verbose", action="store_true", dest="logDebug", default=False, help="Show verbose debugging output")
    lmm_parser.set_defaults(func=lmm_singletrait)

    return inOptions

def checkGenoPhenoFiles(args):
    if not args['genoFile']:
        die("snps file not specified")
    if not args['phenoFile']:
        die("phenotype data not provided")
    if not os.path.isfile(args['genoFile']):
        die("genotype file does not exist: " + args['genoFile'])
    if not os.path.isfile(args['phenoFile']):
            die("phenotype file does not exist: " + args['phenoFile'])

def vca_singletrait(args):
    checkGenoPhenoFiles(args)
    vca.vca_st(args['phenoFile'], args['genoFile'], args['cisregion'], args['kinFile'])

def eqtl_singletrait(args):
    checkGenoPhenoFiles(args)
    vca.eqtl_st(args['phenoFile'], args['genoFile'], args['kinFile'])

def lmm_singletrait(args):
    checkGenoPhenoFiles(args)
    lmm.lmm_singleTrai(args['phenoFile'], args['genoFile'], args['kinFile'], args['outFile'], args['transformation'], args['test'])

def main():
  ''' Command line options '''
  program_version = "v%s" % __version__
  program_build_date = str(__updated__)
  program_version_message = '%%(prog)s %s (%s)' % (program_version, program_build_date)
  program_shortdesc = "The main module1"
  program_license = '''%s
  Created by Rahul Pisupati on %s.
  Copyright 2016 Gregor Mendel Institute. All rights reserved.

  Distributed on an "AS IS" basis without warranties
  or conditions of any kind, either express or implied.
USAGE
''' % (program_shortdesc, str(__date__))

  parser = get_options(program_license,program_version_message)
  args = vars(parser.parse_args())
  setLog(args['logDebug'])
  if 'func' not in args:
    parser.print_help()
    return 0
  try:
    args['func'](args)
    return 0
  except KeyboardInterrupt:
    return 0
  except Exception as e:
    logging.exception(e)
    return 2

if __name__=='__main__':
  sys.exit(main())
