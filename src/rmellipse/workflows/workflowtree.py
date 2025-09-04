import io
from subprocess import Popen, PIPE
from rmellipse.workflows._globals import ProjectSettings
from pathlib import Path
from graphlib import TopologicalSorter
import threading

__all__ = ['DataPointer','Job','WorkflowTree']

class DataPointer(dict):
    """
    Represents an edge in a job tree
    """

    def __init__(self, o: dict | Path):
        dict.__init__(self)
        # assign dictionarry to myself as a shallow copy
        for k, v in o.items():
            self[k] = v

    @property
    def name(self):
        return self['name']

    @property
    def path(self):
        if self['path'] is None:
            return None
        return Path(self['path'])



# executes job and print's it to a file
class Job(dict):
    all_jobs = {}

    def __init__(
        self,
        job_dict: dict,
        project_settings: ProjectSettings
    ):
        """
        Initialize a Job object.

        Parameters
        ----------
        job_dict : dict | io.FileIO | Path
        project_settings : ProjectSettings
            Project settings object.

        Raises
        ------
        TypeError
            _description_
        ValueError
            _description_
        TypeError
            _description_
        """
        if not isinstance(job_dict, dict):
            raise TypeError(f"type {type(job_dict)} invalid to initialize a Job object")

        # assign dictionarry to myself as a shallow object
        for k, v in job_dict.items():
            self[k] = v

        # add self to the dictionairy of jobs
        if self.name in Job.all_jobs:
            raise ValueError(f"Job {self.name} already exists.")
        Job.all_jobs[self.name] = self

        # make a log file path
        config = project_settings
        logfile = config.processlogdir / f"{self.name}.txt"
        self.logfile = logfile
        self.config = config

        # add a thread object to access
        self._thread = threading.Thread(group=None, target=self._run)

        # add something for a return code
        # this is not a part of the "self" so
        # it is not serialized, which is what
        # i want I think because I don't want to
        # be uploading error codes or logs that
        # might contain sensitivit information,
        # and the stderr could be bytes. I only
        # need this at run time.
        self.result = {
            'stderr': None,
            'returncode': None
        }
        # check that every command is a list of strings
        for c in self.commands:
            for ci in c:
                if not isinstance(ci, str):
                    msg = f'Expected str not {type(ci)} for {ci}'
                    msg += f' in job {self.name}'
                    raise TypeError(msg)

    @property
    def thread(self):
        return self._thread

    @property
    def name(self):
        return self["name"]

    @property
    def commands(self):
        return self["commands"]

    def _run(self):
        with io.open(self.logfile, "w") as writer:
            for command in self.commands:
                assert isinstance(command, list)
                for ci in command:
                    if not isinstance(ci, str):
                        msg = f'Expected str not {type(ci)} for {ci}'
                        msg += f' in job {self.name}'
                        raise TypeError(msg)
                kwargs = {
                    'stderr':PIPE,
                    'stdout':PIPE
                }
                with Popen(command,**kwargs, cwd=str(self.config.project_dir.resolve())) as p:
                    while p.poll() is None:
                        for line in p.stdout: # b'\n'-separated lines
                            writer.buffer.write(line)
                    self.result['returncode'] = p.returncode
                    self.result['stderr'] = ''.join([line.decode('utf-8') for line in p.stderr])


class WorkflowTree(dict):#
    def __init__(self, d: dict = None):
        if d is None:
            d = {}
        dict.__init__(self)
        for k in d:
            self[k] = d[k]

        if "jobs" not in self:
            self["jobs"] = {}
        if "data_pointers" not in self:
            self["data_pointers"] = {}
        if "edges" not in self:
            self["edges"] = {}

    @property
    def release(self):
        return self['release']

    @property
    def nodes(self):
        return list(self.jobs.values()) + list(self.data_pointers.values())

    @property
    def jobs(self):
        return self["jobs"]

    @property
    def data_pointers(self):
        return self["data_pointers"]

    @property
    def edges(self):
        return self["edges"]

    @staticmethod
    def _edge_name(o1, o2):
        return o1.name + ", " + o2.name

    def add_data_pointer(self, data_pointer: DataPointer):
        assert isinstance(data_pointer, DataPointer)
        name = data_pointer.name
        if name not in self.data_pointers:
            self.data_pointers[name] = data_pointer
        elif name in self.data_pointers and data_pointer.path is not None:
            # require Data Pointers with a given name
            # to be defined only once
            raise ValueError(f"Cant assign a datapointer twice ({name})")

    def add_job(self, job: Job):
        assert isinstance(job, Job)
        if job.name not in self.jobs:
            self.jobs[job.name] = job
        else:
            raise ValueError(f"Job name must be unique ({job.name})")

    def add_edge(
        self,
        node1: DataPointer | Job,
        node2: DataPointer | Job,
        attrs: dict = None,
    ):
        """
        Add an edge to the workflow tree.

        Edges are always between a DataPointer and a Job.

        Parameters
        ----------
        node1 : DataPointer | Job
            DataPointer or Job
        node2 : DataPointer | Job
            DataPointer or Job
        attrs : dict, optional
            Dict of metadata, by default None
        """
        if attrs is None:
            attrs = {}
        self.edges[self._edge_name(node1, node2)] = {
            "nodes": (node1.name, node2.name)
        }

    def iter_data_pointer_paths(self):
        for k,v in self.data_pointers.items():
            yield v['path']

    def iter_topological_groups(self):
        temp_graph = {}
        for n in self.nodes:
            temp_graph[n.name] = []
        for e in self.edges:
            n1_name, n2_name = self.edges[e]["nodes"]
            temp_graph[n2_name].append(n1_name)
        sorter = TopologicalSorter(temp_graph)
        sorter.prepare()
        while sorter.is_active():
            ready_nodes = sorter.get_ready()
            if ready_nodes:
                yield (ready_nodes)
                sorter.done(*ready_nodes)
            else:
                # This case should ideally not be reached in a valid DAG if is_active() is True
                # unless external conditions prevent nodes from becoming ready.
                print(
                    "No ready nodes, but sorter is still active (potential cycle or external dependency issue)."
                )
                break
