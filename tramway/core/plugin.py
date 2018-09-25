# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


from __future__ import print_function
import importlib
import copy
import os
import re
try:
    fullmatch = re.fullmatch
except AttributeError: # Py2
    fullmatch = re.match
from warnings import warn
import traceback


def list_plugins(dirname, package, lookup={}, force=False, require=None, verbose=False):
    if verbose:
        _pre = 'loading: '
        _post = '...'
        _success = '[done]'
        _failure = '[failed]'
    if not require:
        require = ()
    elif isinstance(require, str):
        require = (require,)
    pattern = re.compile(r'[a-zA-Z0-9].*([.]py)?')
    candidate_modules = set([ os.path.splitext(fn)[0] \
        for fn in os.listdir(dirname) \
        if fullmatch(pattern, fn) is not None ])
    modules = {}
    provided = {}
    for name in candidate_modules:
        path = '{}.{}'.format(package, name)
        # load module
        try:
            module = importlib.import_module(path)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            if verbose:
                print('{}{}{}\t{}'.format(_pre, path, _post, _failure))
                print(traceback.format_exc(), end='')
            continue
        # ensure that all the required attributes are available
        _continue = False
        for required in require:
            if not hasattr(module, required):
                _continue = True
                break
        if _continue:   continue
        elif verbose:
            print('{}{}{}'.format(_pre, path, _post), end='\t')
        # parse setup
        if hasattr(module, 'setup'):
            setup = module.setup
            try:
                name = setup['name']
            except KeyError:
                setup['name'] = name
        else:
            setup = dict(name=name)
        # parse other attributes
        try:
            namespace = module.__all__
        except AttributeError:
            namespace = list(module.__dict__.keys())
        missing = conflicting = None
        warning = []
        for key in lookup:
            if key in setup:
                continue
            ref = lookup[key]
            if isinstance(ref, type):
                matches = []
                for var in namespace:
                    try:
                        ok = issubclass(getattr(module, var), ref)
                    except TypeError:
                        ok = False
                    if ok:
                        matches.append(var)
            else:
                matches = [ var for var in namespace
                    if fullmatch(ref, var) is not None ]
            if matches:
                if matches[1:]:
                    conflicting = key
                    if not force:
                        break
                setup[key] = matches[0]
            else:
                missing = key
                if not force:
                    break
        if conflicting:
            warning.append(("multiple matches in module '{}' for key '{}'".format(path, conflicting), ImportWarning))
            if not force:
                continue
        if missing:
            warning.append(("no match in module '{}' for key '{}'".format(path, missing), ImportWarning))
            if verbose:
                if force:
                    print(_success)
                else:
                    print(_failure)
        elif verbose:
            print(_success)
        for w in warning:
            warn(*w)
        if missing and not force:
            continue
        # register plugin
        plugin = (setup, module)
        if isinstance(name, str):
            modules[name] = plugin
        else:
            names = name
            for name in names:
                modules[name] = plugin
        try:
            provides = setup['provides']
        except KeyError:
            pass
        else:
            if isinstance(provides, str):
                provided[provides] = plugin
            else:
                for _provides in provides:
                    provided[_provides] = plugin
    for provides in provided:
        if provides not in modules:
            modules[provides] = provided[provides]
    return modules


def add_arguments(parser, arguments, name=None):
    translations = []
    for arg, options in arguments.items():
        if not options:
            continue
        long_arg = '--' + arg.replace('_', '-')
        has_options = False
        if isinstance(options, (tuple, list)):
            args = list(options)
            kwargs = args.pop()
            options = {}
        else:
            args = []
            try:
                _parse = options['parse']
            except KeyError:
                pass
            else:
                if callable(_parse):
                    translations.append((arg, _parse))
                    has_options = True
            try:
                kwargs = options['kwargs']
            except KeyError:
                if has_options or not options:
                    continue
                kwargs = options
                options = None # should not be used anymore
            else:
                has_options = True
                try:
                    args = list(options.get('args'))
                except KeyError:
                    pass
        if has_options and options.get('translate', False):
            try:
                _arg = args[1]
            except IndexError:
                _arg = args[0]
            _arg = _arg.lstrip('-').replace('-', '_')
            def _translate(**_kwargs):
                return _kwargs.get(_arg, None)
            translations.append((arg, _translate))
        elif long_arg not in args:
            if args:
                args.insert(1, long_arg)
            else:
                args = (long_arg,)
        try:
            parser.add_argument(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            if name:
                print("WARNING: option `{}` from plugin '{}' ignored".format(arg, name))
            else:
                print("WARNING: option `{}` ignored".format(arg))
    return translations


def short_options(arguments):
    _options = []
    for args in arguments.values():
        if args:
            if isinstance(args, (tuple, list)):
                args = args[0]
            elif 'args' in args:
                args = args['args']
            else:
                continue
            if not isinstance(args, (tuple, list)):
                args = (args,)
            for arg in args:
                if arg[0] != arg[1]:
                    _options.append(arg)
    return _options


class Plugins(object):

    __slots__ = ('modules', 'dirname', 'package', 'lookup', 'force', 'require', 'verbose', 'post_load')

    def __init__(self, dirname, package, lookup={}, force=False, require=None, verbose=False):
        self.modules = None
        self.dirname = dirname
        self.package = package
        self.lookup = lookup
        self.force = force
        self.require = require
        self.verbose = verbose
        self.post_load = None

    def __load__(self):
        if self.modules is None:
            self.modules = list_plugins(
                self.dirname,
                self.package,
                self.lookup,
                self.force,
                self.require,
                self.verbose,
                )
            if self.post_load:
                self.post_load(self)

    def __repr__(self):
        return 'Plugins{{in \'{}\'}}{}'.format(self.package, '{(not loaded)}' if self.modules is None else repr(self.modules))

    def __nonzero__(self):
        self.__load__()
        return self.modules.__nonzero__()

    def __len__(self):
        self.__load__()
        return self.modules.__len__()

    def __iter__(self):
        self.__load__()
        return self.modules.__iter__()

    def __getitem__(self, mod):
        self.__load__()
        return self.modules.__getitem__(mod)

    def __setitem__(self, mod, descr):
        self.__load__()
        self.modules.__setitem__(mod, descr)

    def __missing__(self, mod):
        self.__load__()
        return self.modules.__missing__(mod)

    def keys(self):
        self.__load__()
        return self.modules.keys()

    def values(self):
        self.__load__()
        return self.modules.values()

    def items(self):
        self.__load__()
        return self.modules.items()

    def get(self, mod, default):
        self.__load__()
        return self.modules.get(mod, default)

    def pop(self, mod, default):
        self.__load__()
        return self.modules.pop(mod, default)

    def update(self, plugins):
        self.__load__()
        if not isinstance(plugins, dict):
            raise TypeError('not a `dict`')
        self.modules.update(plugins)


__all__ = [
    'list_plugins',
    'add_arguments',
    'short_options',
    'Plugins',
    ]

