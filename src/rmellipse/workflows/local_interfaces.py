from pathlib import Path
import shutil
import rmellipse.workflows.extras as extras
import hashlib
from rmellipse.workflows._settings import WorkflowConfig, ProjectSettings

__all__ = ['CacheInterface', 'EnvInterface']


class CacheInterface:
    """Interface for a local cache."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)

    def rm_cached_release(self, host: str, workspace: str, release_title: str):
        """
        Remove a cached release.

        Parameters
        ----------
        host : str
            Name of host
        workspace : str
            Name of workspace.
        release_title : str
            Name of release title.
        """
        release_cache = self.get_cached_release_dir(host, workspace, release_title)
        shutil.rmtree(release_cache)

    def get_cached_release_dir(
        self, host: str, workspace: str, release_title: str
    ) -> Path:
        """
        Get a directory in the local cache for a given release.

        Parameters
        ----------
        cache_dir : str
            _description_
        host : str
            _description_
        workspace : str
            _description_
        release : str
            _description_

        Returns
        -------
        Path
            _description_
        """
        normalized_host = Path(host).as_posix()
        sha_1 = hashlib.sha1()
        sha_1.update(normalized_host.encode('utf-8'))
        host_sha = sha_1.hexdigest()
        return self.cache_dir / host_sha / workspace / release_title

    def in_cache(self, host: str, workspace: str, release_title: str) -> bool:
        """
        Check if a release exists in the cache.

        Parameters
        ----------
        host : str
            _description_
        workspace : str
            _description_
        release_title : str
            _description_

        Returns
        -------
        bool
            _description_
        """
        dir = self.get_cached_release_dir(host, workspace, release_title)
        return dir.exists()


class EnvInterface:
    """Interface with an Enviornment."""

    def __init__(self, envdir: Path):
        self.envdir = Path(envdir)
        self.packages_dir = self.envdir / '.packages'
        self.envdir.mkdir(exist_ok=True)
        self.packages_dir.mkdir(exist_ok=True)
        # clean up installations that are broken
        self.rm_broken_packages()

    def rm_broken_packages(self):
        """
        Remove any broken packages in the environment
        """
        packages = self.installed_in_packages()
        registry = self.installed_in_registry()
        missing = set(packages) ^ set(registry)
        for m in missing:
            self.rm_release(m)

    def installed_in_packages(self) -> list[str]:
        """
        Get a list of installed packages

        Returns
        -------
        list[str]
            _description_
        """
        out = []
        for f in self.packages_dir.iterdir():
            if f.name[0] != '.':
                if f.is_dir():
                    out.append(f.name)
        return out

    def installed_in_registry(self) -> list[str]:
        """
        Get a list of releases installed in the registry

        Returns
        -------
        list[str]
            _description_
        """
        out = []
        for f in self.envdir.iterdir():
            if f.name[0] != '.':
                if f.is_dir():
                    out.append(f.name)
        return out

    def release_reg_dir(self, release_title: str):
        """
        Dataset registry directory for a release.

        Parameters
        ----------
        release_title : str
            _description_

        Returns
        -------
        _type_
            _description_
        """
        return self.envdir / release_title

    def release_package_dir(self, release_title: str):
        """
        Package directory for a release.

        Parameters
        ----------
        release_title : str
            _description_

        Returns
        -------
        _type_
            _description_
        """
        return self.packages_dir / release_title

    def in_env(self, release_title: str):
        """
        Determine if a release is installed in the environment.

        Parameters
        ----------
        release_title : str
            _description_

        Returns
        -------
        _type_
            _description_
        """
        return (self.envdir / release_title).exists()

    def rm_release(self, release_title: str):
        """
        Remove a release from the environment.

        Parameters
        ----------
        release_title : str
            _description_
        """
        registry_path = self.envdir / release_title
        full_packages_path = self.packages_dir / release_title
        for dir in [registry_path, full_packages_path]:
            if dir.exists() and dir.is_dir():
                shutil.rmtree(dir)
            elif dir.exists() and dir.is_file():
                dir.unlink()

    @staticmethod
    def with_suffixes_of(path: str | Path, use_suffixes_of: str | Path) -> Path:
        """
        Replace name with the suffixes of use_suffixes_of.

        Parameters
        ----------
        name : str | Path
                string or path
        use_suffixes_of : str | Path
                String or path.

        Returns
        -------
        Path
                Updated path
        """
        suffix = ''.join(Path(use_suffixes_of).suffixes)
        target = Path(path).with_suffix(suffix)
        return target

    def install_files_namespace(
        self, name: str, files: dict[str | Path], resolve_relative_to: Path
    ):
        name = WorkflowConfig._normalize_package_name(name)
        resolve_relative_to = Path(resolve_relative_to)
        # make a place holder directory in the packaegs and tag it
        # so we know its suppose ot be empty down the road
        package_place_holder = self.release_package_dir(name)
        package_place_holder.mkdir(exist_ok=False)
        with open(package_place_holder / '.EMPTYPACKAGE', 'w') as fio:
            fio.write('This file denotes an empty package with only a registry.')

        registry = self.release_reg_dir(name)
        registry.mkdir(exist_ok=False)
        for dname, path in files.items():
            p = Path(path)
            if not p.is_absolute():
                p = resolve_relative_to / p
            self._install_files(registry, dname, p)

    def _install_files(
        self,
        namespace_dir: Path,
        name: str,
        path: str | Path,
    ):
        """
        Install a file into the environment.

        Files are unversioned, and just soft links
        to files that live somewhere else. If path
        is a folder, then the folder structure inside path
        is recreated.

        Parameters
        ----------
        name : str
                New name of file.
        path : str | Path
                Path to file that should be installed.
        """
        path = Path(path).resolve()
        suffix = ''.join(path.suffixes)
        target = (namespace_dir / name).with_suffix(suffix)

        if not path.exists():
            raise FileNotFoundError(path)

        if path.is_file() and path.exists():
            target.symlink_to(path)

        elif path.is_dir() and path.exists():
            target.mkdir(exist_ok=False)
            # make a symbolic link to the cache folder structure
            # in the .packages folder
            extras.sym_link_folder_structure(
                src=path,
                target=target,
            )

    def install_release(
        self,
        release_record: dict,
        workspace: str,
        host: str,
        cache: CacheInterface,
    ):
        """
        Install a release into the environment.

        Parameters
        ----------
        release_record : dict
            _description_
        workspace : str
            _description_
        host : str
            _description_
        cache : CacheInterface
            _description_
        """
        release_title = release_record['title']
        src = cache.get_cached_release_dir(
            host=host,
            workspace=workspace,
            release_title=release_title,
        )
        # rebuild the full package links
        target = self.release_package_dir(release_title)
        target.mkdir(exist_ok=True)

        # make a symbolic link to the cache folder structure
        # in the .packages folder
        extras.sym_link_folder_structure(
            src=src,
            target=target,
        )

        # build registry links
        self.make_registry(
            release_record=release_record,
            cache_release_dir=src,
            use_versioned_name=True,
        )

    def make_registry(
        self,
        release_record: dict,
        cache_release_dir: Path,
        use_versioned_name: bool = True,
    ):
        """
        Make a registry that symbolically links to datasets in the cache.

        Only symbolic links to files are made, directory structures are copied.

        Parameters
        ----------
        release_record : dict
            The record of the of the package being linked too.
        cache : Path
            The cache of the package being linked too. It is assumed
            to already be downloaded.
        env : Path
            The data environment directory.
        use_versioned_name: bool
            If True, used the versioned name when making the registry folder.

        """
        versioned_name = release_record['title']
        versionless_name = release_record['title_versionless']
        datasets = release_record['datasets']
        reg_name = versioned_name
        if not use_versioned_name:
            reg_name = versionless_name

        dataset_registry = self.envdir / reg_name
        dataset_registry.mkdir(exist_ok=False)
        for name, data_pointer in datasets.items():
            src = cache_release_dir / data_pointer
            suffixes = Path(data_pointer).suffixes
            target = dataset_registry / (name + ''.join(suffixes))
            # a dataset may not have been included if it was an intermediate
            # thing
            if src.exists() and src.is_file():
                (target.resolve()).symlink_to(src.resolve())
            # folders are linked out
            elif src.exists() and src.is_dir():
                extras.sym_link_folder_structure(src, target)
