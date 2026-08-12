import os
import sys

this_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bindings'))
PATH_APP = os.path.abspath(os.path.join(this_dir, '..'))
PATH_BINDS = os.path.join(PATH_APP, 'bindings')
PATH_SCRIPTS = os.path.join(PATH_APP, 'scripts')
sys.path.append(PATH_BINDS)
