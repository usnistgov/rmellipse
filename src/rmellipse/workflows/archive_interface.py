from abc import ABC, abstractmethod
from itertools import cycle
from pathlib import Path
from rmellipse.workflows._printtools import braile_load, symbols
from functools import wraps
from typing import Tuple
from yaml import safe_dump, safe_load
import tqdm
import rmellipse.workflows._cdcs_helpers as cdcsh
import concurrent.futures
import rmellipse.workflows.extras as wfh
import rmellipse.workflows._settings as gv
import time
import json
import io
import hashlib
import os
import semver
import keyring
import getpass
import shutil
import inspect
from urllib.parse import urlparse


class ReleaseRecordNotFoundError(Exception):
    """Raise when a record is asked for but doesn't exist."""

    ...


def is_url(input_string):
    try:
        result = urlparse(input_string)
        # A valid URL typically has a scheme (e.g., http, https, ftp)
        # and a network location (e.g., www.example.com)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def chunked_sha(f: io.BytesIO, chunk: int = 2**25):
    """
    Generate a SHA1 for a file in chunks.

    Parameters
    ----------
    f : io.BytesIO

    chunk : int, optional
        How many bytes to chunk at a time, by default 2^20 (~1MB)

    Returns
    -------
    str
        sha1 hash as a string.
    """
    sha256 = hashlib.sha256()
    while True:
        data = f.read(chunk)
        if not data:
            break
        sha256.update(data)
    return sha256.hexdigest()


def get_credentials(host: str, user: str = None, password: str = None):
    """
    Get credentials for a given host and user.

    Prompts user for missing username or password.
    Stores credentials in the system when provided by prompt
    or as function arguments.


    Parameters
    ----------
    host : str
            System host name, typically a URL.
    user : str, optional
            Username, by default None
    password : str, optional
            Password, by default None

    Returns
    -------
    user
            str
    password
            str
    """
    if user is None:
        user = getpass.getpass(prompt=f'{host} user:')

    # prompt for password
    password = keyring.get_password('rme+' + str(host), user)

    if password is None:
        password = getpass.getpass(prompt=f'{host} password:')

    # at this point user and password should both be set
    keyring.set_password('rme+' + str(host), user, password)
    return user, password


def get_interface(
    host: str | Path,
    user: str = None,
    password: str = None,
    resolve_relative_paths_to: str | Path = Path.cwd(),
) -> 'ArchiveInterface':
    """
    Login to an archive, returns an ArchiveInterface.

    Parameters
    ----------
    host : str | Path
        Path to the archive, url or file path.
    user : str, optional
        User name to the archive (if required)
    password : str, optional
        Passwords to the archive (if required).


    Returns
    -------
    ArchiveInterface
        Object that implements the ArchiveInterface.
    """

    if is_url(str(host)):
        user, password = get_credentials(host, user, password)
        return CDCSArchive(host, user, password)

    h = Path(host)
    if not h.is_absolute():
        h = resolve_relative_paths_to / h
    if h.is_dir():
        return FileSystemArchive(h, user)

    raise ValueError(f'Expected URL or directory on {host}')


class ArchiveInterface(ABC):
    """
    Interface for interacting through archives utilized by command line utility functions.

    Any archive should be written into this interface.

    Parameters
    ----------
    ABC :
        _description_

    """

    def __init__(self, host: str | Path, user: str = None, password: str = None):
        self.host = host
        self.user = user

        # wrap functionality expected of interface classes
        self.get_release_records = self._validate_release_record_output(
            self.get_release_records
        )

    def _validate_release_record_output(self, fn):
        """
        Raises an error if the get_release_records function fails.

        Parameters
        ----------
        fn : function
                function to be wrapped.

        Returns
        -------
        dict
                Output of get_release_records
        Raises
        ------
        ReleaseRecordNotFoundError
                No release records were found.
        """

        @wraps(fn)
        def inner(*args, **kwargs):
            available_records = fn(*args, **kwargs)
            if len(available_records) == 0:
                raise ReleaseRecordNotFoundError(
                    inspect.cleandoc(f"""{kwargs['title_versionless']} with version
					{kwargs['version_expressions']} not found in {self.host}
					at workspace {kwargs['workspace']}.""")
                )
            return available_records

        return inner

    @abstractmethod
    def get_release_records(
        self,
        title_versionless: str,
        version_expressions: str = None,
        workspace: str = 'Global Public Workspace',
    ) -> dict:
        """
        Get release records matching a version code.

        Parameters
        ----------
        title_versionless : str
            Title of the release with out the version.
        version_expressions : str
            Version code string (i.e [">=0.1.0","<0.2.0] or ["==0.1.2"])
        workspace : str, optional
            Workspace to look through, by default "Global Public Workspace"

        Returns
        -------
        dict
            Dictionary of release records, keys are the {title}-v{version}
            form
        """
        ...

    def download_release(
        self,
        record: dict,
        workspace: str,
        max_threads: int,
        release_cache_folder: str | Path,
        download_chunk_bytes: int,
        progress_bar: bool = True,
        file_status: bool = True,
    ):
        """
        Download a release from the archive to cache_folder.

        Generates worker threads to download blobs using the download_blob
        function. Provideds progress bars if requested.

        Parameters
        ----------
        record : dict
            _description_
        max_threads : int
            _description_
        progress_bar : bool, optional
            _description_, by default True
        """
        versioned_dataset_name = record['title']

        # recreate the folder structure
        release_cache_folder.mkdir(exist_ok=True, parents=True)
        wfh.folder_structure_from_project_map(record['project'], release_cache_folder)
        # iterate over project mapp, spinning up threads
        # to stream blobs from the server into the local cache
        total_items = 0
        total_bytes = 0
        max_name = 0
        thread_finished_map = {}
        file_sizes = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            for name, cmap in wfh.iter_blobable(record['project']):
                total_bytes += cmap[gv.MAPPING_META_KEYS.BYTES.value]
                pathspec = cmap[gv.MAPPING_META_KEYS.PATHSPEC.value]
                file_sizes[pathspec] = cmap[gv.MAPPING_META_KEYS.BYTES.value]
                max_name = max((max_name), len(name))
                # set a status dict the thread can modify so I can monitor
                # the download status
                target = (
                    release_cache_folder / cmap[gv.MAPPING_META_KEYS.PATHSPEC.value]
                )

                # download if file size is > 0
                if file_sizes[pathspec] > 0:
                    total_items += 1
                    thread_finished_map[pathspec] = {
                        'size': 0,
                        'finished': False,
                        'expected': file_sizes[pathspec],
                    }
                    executor.submit(
                        self.download_blob,
                        blob_pid=cmap[gv.MAPPING_META_KEYS.BPID.value],
                        target_file=target,
                        update_dict=thread_finished_map[pathspec],
                        expected_size=file_sizes[pathspec],
                        release_record=record,
                        workspace=workspace,
                        chunk_size=download_chunk_bytes,
                    )
                #  0 bytes files, just make an empty file.
                else:
                    with open(target, 'w'):
                        pass

        # monitor each uploading thread
        # and make a progress bar that compares the total expected
        # file size
        finished_count = 0
        with tqdm.tqdm(
            total=total_bytes,
            desc=f'{versioned_dataset_name}',
            unit='B',
            unit_scale=True,
        ) as pbar:
            while finished_count < total_items:
                current_size = 0
                finished_count = 0
                new_line_count = 0
                msg = ''
                for pathspec, status in thread_finished_map.items():
                    if status['finished']:
                        finished_count += 1
                        if not status['success']:
                            raise Exception(status['error'])
                    else:
                        msg += f'{pathspec} | {status["size"]}'
                        new_line_count += 1
                    current_size += status['size']
                pbar.update(current_size)
                # print('-----')
                # print(msg, end = '')
                time.sleep(0.025)
        executor.shutdown(wait=True)
        # Save the record inside the folder
        # save the record to the folder structure
        dotrme = release_cache_folder / gv.DOTRME_DIRNAME
        dotrme.mkdir(parents=True, exist_ok=True)
        with open(dotrme / 'record.yml', 'w') as f:
            safe_dump(record, f)

    @abstractmethod
    def download_blob(
        self,
        blob_pid: str,
        target_file: Path,
        update_dict: dict,
        expected_size: int,
        release_record=dict,
        workspace=str,
        chunk_size=2**25,
    ):
        """
        Stream a blob stored in the archive to your local computer.

        Parameters
        ----------
        blob_pid : str
            PID of the blob.
        target_file : Path
            Target file in local path.
        update_dict : dict
            Empty dictionary, updated with the
        expected_size : int
            _description_
        chunk_size : _type_, optional
            _description_, by default 2**25

        Returns
        -------
        _type_
            _description_
        """

    @abstractmethod
    def process_and_upload_blob(
        self,
        posix_rel_path,
        working_dir,
        release_title_versionless,
        release_title,
        workspace_title,
        chunk_size,
        verbose=False,
    ) -> Tuple[str, int]:
        """
        Upload a blob to the archive.

        Returns the blob_PID and the number of bytes.

        Parameters
        ----------
        posix_rel_path : _type_
                _description_
        working_dir : _type_
                _description_
        release_title_versionless : _type_
                _description_
        release_title : _type_
                _description_
        workspace_title : _type_
                _description_
        chunk_size : _type_
                _description_
        verbose : bool, optional
                _description_, by default False

        Returns
        -------
        blob_pid: str
            The  PID of the blob (i.e. file) that was uploaded
        nbytes: int
            The size of the uploaded in bytes.
        """
        ...

    def _upload_single_blob(
        self,
        blob_map: dict,
        project_directory: Path,
        no_blobs: bool,
        thread_finished_map: dict,
        release_title: str,
        release_title_versionless: str,
        workspace_title: str,
        chunk_size: int,
    ):
        """
        Thread process envoked by upload blobs.

        Parameters
        ----------
        cmap : dict
            _description_
        curator : cdcsh.CachedCurator
            _description_
        project_directory : Path
            _description_
        no_blobs : bool
            _description_
        thread_finished_map : dict
            _description_
        release_title : str
                Name of the release
        chunk_size: int
                Max size of chunks
        """
        # move cursor to beginning of list
        if not no_blobs:
            try:
                blob_pid, nbytes = self.process_and_upload_blob(
                    posix_rel_path=blob_map['/path/'],
                    working_dir=project_directory,
                    release_title_versionless=release_title_versionless,
                    release_title=release_title,
                    workspace_title=workspace_title,
                    chunk_size=chunk_size,
                    verbose=False,
                )
            except Exception as e:
                thread_finished_map[blob_map[gv.MAPPING_META_KEYS.PATHSPEC.value]][
                    'errors'
                ] = e
        else:
            blob_pid = 'NONE'
            nbytes = 0
        # assign the blob id to the mapping of the blob
        blob_map[gv.MAPPING_META_KEYS.BPID.value] = blob_pid
        blob_map[gv.MAPPING_META_KEYS.BYTES.value] = nbytes

        # update the thread finished portion
        thread_finished_map[blob_map[gv.MAPPING_META_KEYS.PATHSPEC.value]][
            'complete'
        ] = True

    def upload_blobs_and_update_mapping(
        self,
        release_title: str,
        release_title_versionless: str,
        workspace: str,
        project_mapping: dict,
        project_directory: str | Path,
        max_threads: int,
        chunk_size: int,
        no_blobs: bool = False,
    ):
        """
        Upload blobs to the archive and update the project mapping.

        The project mapping metadata for each blob is updated with information
        that is only determined at upload time (i.e. the PID, bytes). Should
        be called before upload release record.

        Parameters
        ----------
        project_mapping : dict
            Project mapping dictionary.
        project_directory : str | Path
            Root directory of the project.
        max_threads : int
            Max threads for upload processes. Each blob
            gets its own process.
        chunk_size : int
                Max size of chunks for uploading
        no_blobs : bool, optional
            Dont upload blobs, by default False, for
            debugging purposes only.
        """
        total = 0
        max_name = 0
        thread_finished_map = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            for name, blob_map in wfh.iter_blobable(project_mapping):
                total += 1
                max_name = max((max_name), len(name))
                thread_finished_map[blob_map[gv.MAPPING_META_KEYS.PATHSPEC.value]] = {
                    'complete': False,
                    'errors': None,
                }

                executor.submit(
                    self._upload_single_blob,
                    blob_map,
                    project_directory,
                    no_blobs,
                    thread_finished_map,
                    release_title,
                    release_title_versionless,
                    workspace,
                    chunk_size,
                )

            print('total blobable items: ', total)
            load_sym = cycle(braile_load)
            finished_sym = symbols.CHECK
            finished_count = 0
            cursor_count = 0
            # monitor each uploading thread
            while finished_count < total:
                print('\033[F' * cursor_count, end='')
                msg = ''
                ongoing_sym = next(load_sym)
                finished_count = 0
                cursor_count = 0
                for i, (pathspec, fmap) in enumerate(thread_finished_map.items()):
                    if fmap['complete']:
                        finished_count += 1
                        sym = finished_sym
                    else:
                        sym = ongoing_sym
                    # print ongoing upload if less then 10
                    if i < 10:
                        cursor_count += 1
                        msg += f'{sym} | {pathspec.ljust(max_name)}\n'

                if total > 10:
                    msg += 'other processes hidden ...\n'
                    cursor_count += 1

                msg = f'completed {finished_count}/{total}\n' + msg
                cursor_count += 1
                cursor_count += 1
                print(msg)

                time.sleep(0.1)

            executor.shutdown(wait=True)
            for i, (pathspec, fmap) in enumerate(thread_finished_map.items()):
                if fmap['errors']:
                    raise fmap['errors']

    @abstractmethod
    def upload_release_record(title: str, release_record: dict) -> str:
        """
        Upload a release record.

        The record should be assigned a PID during the upload process.

        Parameters
        ----------
        title : str
            Title of the release with version code.
        release_record : dict
            Releae record dictionary.

        Returns
        -------
        str
            PID of uploaded object.
        """
        ...


# %% Archive
class CDCSArchive(ArchiveInterface):
    """Archive Interface for a CDCS instance."""

    def __init__(self, host: str, user: str, password: str = None):
        ArchiveInterface.__init__(self, host, user, str)
        self.curator = cdcsh.login(
            hostname=host, username=user, password=password, verbose=False
        )
        self.supports_repeat_releases = True

    def get_release_records(
        self,
        title_versionless: str,
        version_expressions: list[str],
        workspace='Global Public Workspace',
    ) -> dict:
        """
        Get release records matching a version code.

        Parameters
        ----------
        title_versionless : str
            Title of the release with out the version.
        version_expressions : str
            Version code string (i.e [">=0.1.0","<0.2.0] or ["==0.1.2"])
        workspace : str, optional
            Workspace to look through, by default "Global Public Workspace"

        Returns
        -------
        dict
            Dictionary of release records, keys are the {title}-v{version}
            format sorted from oldest to most recent version following semver.
        """
        # grab the available versions for that dataset
        # sorted by release
        mongoquery = {
            'title_versionless': title_versionless,
        }
        if version_expressions is None:
            version_expressions = ['>=0.0.0']

        releases = self.curator.query(
            template='Release',
            parse_dates=True,
            current=False,
            mongoquery=mongoquery,
            progress_bar=False,
        )

        # in case of repeat releases, sort by creation date
        # so the newest release of a particular version gets
        # inserted last, and is the only one kept.
        releases = releases.sort_values(by='creation_date').reset_index()
        # make a dictionary look of releases with satsified version codes
        # sorted from oldest to most recent
        out = {
            releases.title[i]: json.loads(releases.content[i])
            for i in range(len(releases))
        }

        # filter out any versions that don't match semver expression
        titles = list(out.keys())
        keep = []
        for t in titles:
            v = out[t]['version']
            if all([semver.match(v, exp) for exp in version_expressions]):
                keep.append(t)
        out = {k: out[k] for k in keep}

        # sory by version comparison
        def get_version(k):
            return semver.Version.parse(out[k]['version'])

        new_keys = list(out.keys())
        new_keys.sort(key=get_version)
        out = {nk: out[nk] for nk in new_keys}
        return out

    def download_blob(
        self,
        blob_pid: str,
        target_file: str | Path,
        update_dict: dict,
        expected_size: int,
        release_record=dict,
        workspace=str,
        chunk_size: int = 2**25,
    ):
        """
        Download a blob from a PID to it's a local file.

        Parameters
        ----------
        blob_pid : str
            PID of blob, should be url.
        target_file : str | Path
            Target path to download to.
        update_dict : dict
            Dictionary with {'size':0,'finished':false},
            used to monitor the download process when spun up
            into threads.
        expected_size : int
            Expected size of the blob in bytes.
        chunk_size : int, optional
            Chunk size for downloading, by default 2**25
        """
        cdcsh.stream_blob_to_file(
            self.curator,
            blob_pid=blob_pid,
            target_file=target_file,
            update_dict=update_dict,
            expected_size=expected_size,
            chunk_size=chunk_size,
        )

    def process_and_upload_blob(
        self,
        posix_rel_path: Path,
        working_dir: Path,
        release_title: str,
        release_title_versionless: str,
        workspace_title: str,
        chunk_size: int,
        verbose: bool = False,
    ) -> tuple[str]:
        """
        Upload a file to a CDCS workspace.

        Parameters
        ----------
        posix_rel_path : Path
            _description_
        working_dir : Path
                Working directory of release, from with all paths
                are relative.
        release_title: str
                Name of the release with version code. Isn't required
                for the CDCS archive, but included for interface
                compatability.
        release_title_versionless: str
                Name of the release without the version code.
        chunk_size : int
                Chunking size. Not used
        workspace_title: str
                Name of the workspace to upload to.

        Returns
        -------
        blob_pid:
            blob_id
        nbytes:
            size of file in bytes
        """

        # first get a hash of the blob, see
        # if it already exists
        # if it does, then just return that pid
        nbytes = os.path.getsize(working_dir / posix_rel_path)

        if nbytes == 0:
            blob_pid = None
        else:
            with open(working_dir / posix_rel_path, 'rb') as f:
                sha1 = chunked_sha(f)
                try:
                    blob_pid = self.curator.get_sha1_pid(sha1, verbose=verbose)
                    if verbose:
                        print('File already exists.')
                except cdcsh.HashNotInDatabaseError:
                    if verbose:
                        print('SHA1 not in database, uploading.')
                    blob_pid = None

        # if a pid wasn't assigned, it means it wasn't
        # found in the database and needs to be
        # uploaded and assigned a PID
        if blob_pid is None and nbytes > 0:
            with open(working_dir / posix_rel_path, 'rb') as f:
                # TODO: This should probably be chunked
                bcontent = f.read()

                # blob id is the download url
                fname = Path(posix_rel_path).name
                blob_id = self.curator.upload_blob(
                    filename=fname,
                    blobbytes=bcontent,
                    workspace=workspace_title,
                    verbose=verbose,
                )
                blob_id = blob_id.split('/')[-2]
                blob_meta = self.curator.get_blob(id=blob_id)
                blob_pid = blob_meta.pid

                # generate a blob metadata record
                blob_meta_rec = {'sha1': sha1, 'blob/PID/': blob_pid, '/bytes/': nbytes}
                # make a record of metadata
                meta_record = cdcsh.upload_record(
                    curator=self.curator,
                    title=f'{fname}-meta',
                    template_title='BlobMetadata',
                    content=blob_meta_rec,
                    workspace_title=workspace_title,
                )
                meta_record_id = meta_record.json()['id']
                meta_record_data = json.loads(meta_record.json()['content'])

                # assign metadata to the blob
                rest_url = f'/rest/blob/{blob_id}/metadata/{meta_record_id}/'
                response = self.curator.post(rest_url)
                cdcsh.raise_from_status_code(response)

        return blob_pid, nbytes

    def upload_release_record(self, release_record: dict, workspace: str) -> str:
        """
        Upload a release record.

        The record should be assigned a PID during the upload process.

        Parameters
        ----------
        title : str
            Title of the release with version code.
        release_record : dict
            Releae record dictionary.

        Returns
        -------
        str
            PID of uploaded object.
        """
        response = cdcsh.upload_record(
            curator=self.curator,
            title=release_record['title'],
            template_title='Release',
            content=release_record,
            workspace_title=workspace,
        )

        pid = json.loads(response.json().get('content'))['/PID/']
        return pid


# %% Interface for a file system archive
class FileSystemArchive(ArchiveInterface):
    """
    Archive Interface for a file system archive.

    File system archive, which is an archive stored just
    in a directory with a standard layout. Access permissions
    are based on file system persmissions in the system.

    """

    def __init__(self, host: str, user: str):
        ArchiveInterface.__init__(self, host)
        self.archive_path = Path(host).resolve()
        if not (self.archive_path / '.rmearchive').exists():
            raise FileNotFoundError(f'{str(self.archive_path)} is not an rme archive.')
        # print(self.archive_path)
        # my host is a path, so should be made into a posix path
        self.host = Path(self.host).as_posix()

    def get_release_records(
        self,
        title_versionless: str,
        version_expressions: list[str],
        workspace='Global Public Workspace',
    ) -> dict:
        """
        Get release records matching a version code.

        Parameters
        ----------
        title_versionless : str
            Title of the release with out the version.
        version_expressions : str
            Version code string (i.e [">=0.1.0","<0.2.0] or ["==0.1.2"])
        workspace : str, optional
            Workspace to look through, by default "Global Public Workspace"

        Returns
        -------
        dict
            Dictionary of release records, keys are the {title}-v{version}
            format sorted from oldest to most recent version following semver.
        """
        if version_expressions is None:
            version_expressions = ['>=0.0.0']
        release_folder = self.archive_path / workspace / title_versionless
        if not release_folder.exists():
            return {}
        # these are all the release records in the archive
        record_files = [
            file
            for file in release_folder.iterdir()
            if title_versionless in file.name and file.is_file()
        ]
        records = []
        for rf in record_files:
            with open(rf, 'r') as fio:
                records.append(safe_load(fio))
        # these are all the
        # filter out any versions that don't match semver expression
        keep = {}
        for r in records:
            v = r['version']
            title = r['title']
            if all([semver.match(v, exp) for exp in version_expressions]):
                keep[title] = r
        return keep

    def download_blob(
        self,
        blob_pid: str,
        target_file: str | Path,
        update_dict: dict,
        expected_size: int,
        release_record: dict,
        workspace: str,
        chunk_size: int = 2**25,
    ):
        """
        Download a blob from a PID to it's a local file.

        Parameters
        ----------
        blob_pid : str
            PID of blob, should be url.
        target_file : str | Path
            Target path to download to.
        update_dict : dict
            Dictionary with {'size':0,'finished':false},
            used to monitor the download process when spun up
            into threads.
        expected_size : int
            Expected size of the blob in bytes.
        release_record: dict
                Full release record.
        workspace: str,
                Workspace to look through
        chunk_size : int, optional
            Chunk size for downloading, by default 2**25
        """
        # with open(target_file, 'wb') as out_file:
        # 	content = requests.get(blob_pid, stream=True).content
        # 	out_file.write(content)
        try:
            title = release_record['title']
            versionless = release_record['title_versionless']
            release_folder = self.archive_path / workspace / versionless
            sha = blob_pid
            blob_file = release_folder / 'blobs' / sha[0:2] / sha
            shutil.copy(blob_file, target_file)
            update_dict['success'] = True
            update_dict['size'] = expected_size
        except Exception as e:
            update_dict['error'] = e
            update_dict['success'] = False
        update_dict['finished'] = True

    def process_and_upload_blob(
        self,
        posix_rel_path: Path,
        working_dir: Path,
        release_title_versionless: str,
        release_title: str,
        workspace_title: str,
        chunk_size: int,
        verbose: bool = False,
    ) -> tuple[str]:
        """
        Upload a file to a CDCS workspace.

        Parameters
        ----------
        posix_rel_path : Path
                _description_
        working_dir : Path,
                Working directory of the project
        release_title_versionless: str
                Name of release without version code
        release_title: Str
                Name of release with version
        verbose : bool,
                Print information
        workspace_title: str, optional
                Name of the workspace.
        chunk_size: int
                Size of upload in chunks.

        Returns
        -------
        blob_pid:
                blob_id
        nbytes:
                size of file in bytes
        """
        # first get a hash of the blob, see
        # if it already exists
        # if it does, then just return that pid
        nbytes = os.path.getsize(working_dir / posix_rel_path)
        with open(working_dir / posix_rel_path, 'rb') as f:
            sha = chunked_sha(f, chunk=chunk_size)
        blob_pid = sha

        # make a blobs folder if it isn't there
        release_folder = self.archive_path / workspace_title / release_title_versionless
        blobs_folder = release_folder / 'blobs'

        # make a folder to store blobs with the
        # same first 2 characters of the SHA1 hash
        hash_starter = sha[:2]
        this_blob_folder = blobs_folder / hash_starter
        this_blob_folder.mkdir(exist_ok=True, parents=True)
        # if the hash is already in the folder, return
        this_blob_path = this_blob_folder / sha

        # copy file to archive if the hash isn't already stored
        if not this_blob_path.exists():
            shutil.copy(working_dir / posix_rel_path, this_blob_path)

        return blob_pid, nbytes

    def upload_release_record(self, release_record: dict, workspace_title: str) -> str:
        title = release_record['title']
        versionless = release_record['title_versionless']
        release_folder = self.archive_path / workspace_title / versionless
        release_file = release_folder / (title + '.yml')
        with open(release_file, 'x') as fio:
            safe_dump(release_record, fio)
        return release_file
