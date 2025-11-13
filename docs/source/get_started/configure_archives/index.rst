Make an Archive
===============

To start a new archive, just make an empty folder
anywhere and add an empty file called ``.rmearchive``
inside of it.

Set User Defaults
-----------------

Archive settings can be configured at the user
level or the project level. For the project level,
include a file called ``rme.yml`` in the projects root
directory. For the user level, add it to ``~/.rmellipse/rme.yml``
in the user's home folder.

Archive settings are overloaded based
on discovery priority order, from least to greatest that order is:

1. User settings
2. Project settings




Set archives by adding the following to ``rme.yml``:

.. code-block:: yaml

    archives:
      archive-key1: # short hand name for identifying archives
          host: path/to/archive1 # Path to archive
          default: True # flags the archive as the default when pulling in dependencies (optional)
          user: user # user name (optional)

      # you can have multiple archives configured
      archive-key2:
          host: path/to/archive2
          user: user

Note, a password can be provided by adding a ``password`` field to the archive key, however this is insecure.
Be careful when doing this not to accidentally commit secrets to git, or publish them to the archive.

Passwords are stored and accessed when used with keyring, and prompted for if the password
hasn't been stored yet. It's recommended just to provide a username for each host, and not
to keep it in at the project level unless it is ignored from version control and the release.
