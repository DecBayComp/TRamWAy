# -*- coding: utf-8 -*-

# Copyright © 2020-2021, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


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
        """
        account information in the following format: *username@hostname*
        """
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
        """
        read-only property that requests a password from the user if a password is required.

        set attribute `_password` to explicitly pass a password.
        Note however plain text passwords are security breaches.
        """
        if self._password is None:
            if __user_interaction__ is True:
                import getpass
                self._password = getpass.getpass(self.host+"'s password: ")
            else:
                raise RuntimeError('a password is required')
        return self._password
    @property
    def options(self):
        """
        keyword arguments passed to `paramiko.client.SSHClient.connect`.

        additional supported options are:

        * *allow_password* (bool): makes `connect` crash if a password is required
                and *allow_password* is ``False``

        """
        return self._options
    def connect(self):
        try:
            import paramiko
        except ImportError:
            raise ImportError('package paramiko is required')
        try:
            user, host = self.host.split('@')
        except ValueError:
            user, host = None, self.host
        self._conn = paramiko.SSHClient()
        self._conn.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._conn.load_system_host_keys()
        allow_password = self.options.pop('allow_password', None)
        try:
            self._conn.connect(host, 22, user, **self.options)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            if allow_password is False:
                raise
            else:
                self._conn.connect(host, 22, user, self.password, **self.options)
        if allow_password is not None:
            self.options['allow_password'] = allow_password
    def exec(self, cmd, shell=False, logger=None):
        """
        runs command *cmd* on the remote host.

        if some executable is reported missing, set ``shell=True``.
        """
        if shell:
            cmd = 'bash -l -c "{}"'.format(cmd.replace('"',r'\"'))
        if logger is not None:
            logger.debug('running command: '+cmd)
        _in, _out, _err = self.connection.exec_command(cmd)
        out = _out.read()
        if out and not isinstance(out, str):
            out = out.decode('utf-8')
        err = _err.read()
        if err and not isinstance(err, str):
            err = err.decode('utf-8')
        return out, err
    def put(self, src, dest, confirm=False):
        """
        uploads *src* to remote file *dest*.

        see also `paramiko.sftp_client.SFTPClient.put`.
        """
        return self.sftp_client.put(src, dest, confirm=confirm)
    def get(self, target, dest):
        """
        downloads *target* as local file *dest*.

        see also `paramiko.sftp_client.SFTPClient.put`.
        """
        if target.startswith('~/'):
            target = target[2:]
        dest = os.path.expanduser(dest)
        self.sftp_client.get(target, dest)
    def download_if_missing(self, target, target_url, logger=None):
        """
        downloads *target_url* if *target* is missing on the remote host.

        note: command *wget* is run from the remote host.
        """
        try:
            info = self.sftp_client.lstat(target)
        except:
            info = None
        if info is None:
            if logger is not None:
                logger.info('downloading {}...'.format(target_url))
            out, err = self.exec('wget '+target_url)
            if err and logger is not None:
                logger.error(err)
            return not err
    def close(self):
        if self._conn is not None:
            self._conn.close()


__all__ = ['Client']

