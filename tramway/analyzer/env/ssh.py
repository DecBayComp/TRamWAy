# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


try:
    import paramiko
except ImportError:
    raise ImportError('package paramiko is required')
import os.path
from tramway.core.rc import __user_interaction__


class Client(object):
    """
    encapsulates the low-level API of Paramiko,
    and merely simplifies authentification.
    """
    __slots__ = ('_host','_conn','_sftp_client','_password','_options')
    def __init__(self, host=None, **options):
        self._host = host
        self._conn = None
        self._sftp_client = None
        self._password = None
        self._options = options
    @property
    def host(self):
        return self._host
    @host.setter
    def host(self, addr):
        self._host = addr
    @property
    def connection(self):
        if self._conn is None:
            self.connect()
        return self._conn
    @property
    def sftp_client(self):
        if self._sftp_client is None:
            self._sftp_client = self.connection.open_sftp()
        return self._sftp_client
    @property
    def password(self):
        if self._password is None:
            if __user_interaction__ is True:
                import getpass
                self._password = getpass.getpass(self.host+"'s password: ")
            else:
                raise RuntimeError('a password is required')
        return self._password
    @property
    def options(self):
        return self._options
    def connect(self):
        user, host = self.host.split('@')
        self._conn = paramiko.SSHClient()
        self._conn.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._conn.load_system_host_keys()
        allow_password = self.options.pop('allow_password', None)
        try:
            self._conn.connect(host, 22, user, **self.options)
        except:
            if allow_password is False:
                raise
            else:
                self._conn.connect(host, 22, user, self.password, **self.options)
        if allow_password is not None:
            self.options['allow_password'] = allow_password
    def exec(self, cmd, shell=False, logger=None):
        if shell:
            cmd = 'bash -l -c "{}"'.format(cmd.replace('"',r'\"'))
        if logger is not None:
            logger.info('running command: '+cmd)
        _in, _out, _err = self.connection.exec_command(cmd)
        out = _out.read()
        if out and not isinstance(out, str):
            out = out.decode('utf-8')
        err = _err.read()
        if err and not isinstance(err, str):
            err = err.decode('utf-8')
        return out, err
    def put(self, src, dest, confirm=False):
        return self.sftp_client.put(src, dest, confirm=confirm)
    def get(self, target, dest):
        if target.startswith('~/'):
            target = target[2:]
        dest = os.path.expanduser(dest)
        self.sftp_client.get(target, dest)
    def close(self):
        if self._conn is not None:
            self._conn.close()


__all__ = ['Client']

