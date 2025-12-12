import io
import logging
from subprocess import Popen, PIPE
from rmellipse.workflows._settings import ProjectSettings, WorkflowConfig
from pathlib import Path
from graphlib import TopologicalSorter
from rmellipse.workflows._printtools import Colors, symbols, cstr, cprint, PrintManager
from itertools import cycle
import threading
import time
import psutil
import multiprocessing.dummy
import json
from concurrent.futures import ThreadPoolExecutor
import traceback
import sys

__all__ = [
	'DataPointer',
	'Job',
	'WorkflowTree',
	'execute_concurrent_jobs',
	'validate_datapointers',
]


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

	def __init__(self, job_dict: dict, project_settings: ProjectSettings):
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
			raise TypeError(f'type {type(job_dict)} invalid to initialize a Job object')

		# assign dictionarry to myself as a shallow object
		for k, v in job_dict.items():
			self[k] = v

		# add self to the dictionairy of jobs
		if self.name in Job.all_jobs:
			raise ValueError(f'Job {self.name} already exists.')
		Job.all_jobs[self.name] = self

		# make a log file path
		config = project_settings
		logfile = config.processlogdir / f'{self.name}.txt'
		self.logfile = logfile
		self.config = config

		# add something for a return code
		# this is not a part of the "self" so
		# it is not serialized, which is what
		# i want I think because I don't want to
		# be uploading error codes or logs that
		# might contain sensitivit information,
		# and the stderr could be bytes. I only
		# need this at run time.
		self.result = {
			'PID': None,
			'stderr': None,
			'returncode': None,
			'mem_usage': 6.9,
			'mem_percent': 0.0,
			'cpu_percent': 0.0,
		}
		self.started = False

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
		return self['name']

	@property
	def commands(self):
		return self['commands']

	def run(self):
		try:
			self.started = True
			with io.open(self.logfile, 'w') as writer:
				for command in self.commands:
					if not isinstance(command, str):
						msg = f'Expected str not {type(command)} for {command}'
						msg += f' in job {self.name}'
						raise TypeError(msg)
					kwargs = {'stderr': PIPE, 'stdout': PIPE}
					with Popen(
						command,
						**kwargs,
						shell=True,
						cwd=self.config.project_dir.resolve(),
					) as p:
						while p.poll() is None:
							try:
								process = psutil.Process(p.pid)
								with process.oneshot():
									try:
										memory = process.memory_info()
										self.result['PID'] = p.pid
										self.result['mem_usage'] = memory.rss
										self.result['mem_percent'] = (
											process.memory_percent()
										)
										self.result['cpu_percent'] = (
											process.cpu_percent()
										)
									except psutil.NoSuchProcess:
										pass
							except psutil.NoSuchProcess:
								pass

							for line in p.stdout:  # b'\n'-separated lines
								writer.buffer.write(line)

						self.result['returncode'] = p.returncode
						self.result['stderr'] = ''.join(
							[line.decode('utf-8') for line in p.stderr]
						)
						if self.result['returncode'] != 0:
							break
		except Exception as e:
			self.result['returncode'] = 1
			msg = traceback.format_exc()
			self.result['stderr'] = msg + '\n Error in job execution:' + str(e)


class WorkflowTree(dict):  #
	def __init__(self, d: dict = None):
		if d is None:
			d = {}
		dict.__init__(self)
		for k in d:
			self[k] = d[k]

		if 'jobs' not in self:
			self['jobs'] = {}
		if 'data_pointers' not in self:
			self['data_pointers'] = {}
		if 'edges' not in self:
			self['edges'] = {}

	@property
	def release(self):
		"""Release information about the workflow tree."""
		return self['release']

	@property
	def nodes(self):
		"""Access all the nodes in the graph representation."""
		return list(self.jobs.values()) + list(self.data_pointers.values())

	@property
	def jobs(self):
		"""All the job nodes."""
		return self['jobs']

	@property
	def data_pointers(self):
		"""All the data_pointer nodes."""
		return self['data_pointers']

	@property
	def edges(self):
		"""All the edges in the tree."""
		return self['edges']

	@staticmethod
	def _edge_name(o1: DataPointer | Job, o2: DataPointer | Job):
		"""Format an edge name."""
		return o1.name + ', ' + o2.name

	def add_data_pointer(self, data_pointer: DataPointer):
		"""Add a data pointer to the tree."""
		assert isinstance(data_pointer, DataPointer)
		name = data_pointer.name
		if name not in self.data_pointers:
			self.data_pointers[name] = data_pointer

	def add_job(self, job: Job):
		"""Add a job to the tree."""
		assert isinstance(job, Job)
		if job.name not in self.jobs:
			self.jobs[job.name] = job
		else:
			raise ValueError(f'Job name must be unique ({job.name})')

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
		self.edges[self._edge_name(node1, node2)] = {'nodes': (node1.name, node2.name)}

	def iter_data_pointer_paths(self):
		"""
		Iterate over all the data_pointer paths.

		Yields
		------
		Path
		    Path to data_pointers
		"""
		for k, v in self.data_pointers.items():
			yield v['path']

	def iter_topological_groups(self):
		"""
		Iterate over groups in a topologial sorting of jobs and datasets.

		Yields
		------
		list
		    List of nodes in the next level of the topological sorting.
		"""
		temp_graph = {}
		for n in self.nodes:
			temp_graph[n.name] = []
		for e in self.edges:
			n1_name, n2_name = self.edges[e]['nodes']
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
					'No ready nodes, but sorter is still active (potential cycle or external dependency issue).'
				)
				break

	@classmethod
	def from_workflow_config(
		cls,
		project_settings: ProjectSettings,
		wf_config: WorkflowConfig,
	):
		# set up the workflow tree based on the config file
		wf_tree = WorkflowTree()
		for job_name in wf_config.jobs:
			wf_tree._add_job_from_config(project_settings, wf_config, job_name)
		wf_tree['release'] = wf_config.release
		return wf_tree

	def _add_job_from_config(
		self,
		project_settings: ProjectSettings,
		wf_config: WorkflowConfig,
		job_name: str,
	):
		"""
		Add a job from a workflow config.

		Helper method for the from_workflow_config
		class method.

		Parameters
		----------
		project_settings : ProjectSettings
		    Project settings.
		wf_config : WorkflowConfig
		    WorkflowSettings.
		job_name : str
		    Name of job.
		Raises
		------
		KeyError
		    _description_
		ValueError
		    _description_
		ValueError
		    _description_
		"""
		# add a job
		k = job_name
		job_dict = wf_config.jobs[job_name]
		job_dict.update({'name': k})
		# print(job_dict)

		if 'for each' in job_dict:
			job_dict_str = json.dumps(job_dict)
			index = '{' + job_dict['for each']['index'] + '}'
			for i, iterator in enumerate(job_dict['for each']['in']):
				job_dict_i = job_dict_str.replace(index, iterator)
				job_dict_i = json.loads(job_dict_i)
				job_dict_i['name'] += f'-{i}'
				self._add_job_from_dict(job_dict_i, project_settings)

		else:
			self._add_job_from_dict(job_dict, project_settings)

	def _add_job_from_dict(
		self,
		job_dict,
		project_settings,
	):
		j = Job(job_dict, project_settings)
		# print(command)
		self.add_job(j)
		try:
			outputs = job_dict.pop('outputs')
		except KeyError:
			outputs = []

		try:
			inputs = job_dict.pop('inputs')
		except KeyError:
			inputs = []

		# connect job to inputs
		for path in inputs:
			input = DataPointer({'name': path, 'path': path})
			self.add_data_pointer(input)
			self.add_edge(input, j)

		# connect job to inputs
		for path in outputs:
			output = DataPointer({'name': path, 'path': path})
			self.add_data_pointer(output)
			self.add_edge(j, output)


# %% Functions that utilize Jobs, Datapointers, and Workflow trees
def execute_concurrent_jobs(jobs: list[Job], level: int, max_threads: int = None):
	"""
	Execute a set of concurrent jobs in wft.

	Parameters
	----------
	jobs : list[job]
	    List of jobs to execute concurrently.
	group_name : str
	    Name of group, used for printing.
	max_threads : int
	    Max number of threads (i.e. subprocesses) that
	    can be spun up at a time.

	Raises
	------
	TypeError
	    If non-jobs are passed.
	Exception
	    When a job fails.
	"""
	max_n = max(len(j.name) for j in jobs)

	row_str = '  {} | {} | mem: {:5.2f}% | cpu: {:5.2f}%\n'
	header_str = '{}: Jobs | sys mem: {:5.2f}% | sys cpu: {:5.2f}%\n'
	# if on a job level, just execute each job

	completed = []
	failed = set()
	load_syms = cycle(['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'])
	s = next(load_syms)

	def run_job(job):
		job.run()

	pm = PrintManager()
	if max_threads is None:
		max_threads = multiprocessing.cpu_count() - 1

	with ThreadPoolExecutor(max_workers=max_threads) as pool:
		workers = []
		for i, cj in enumerate(jobs):
			worker = pool.submit(run_job, cj)
			workers.append(worker)
			# except TypeError as e:
			#     raise SystemExit(f'Failed to start {cj.name}: ') from e
			completed.append(False)
			# print(f'  {s} | {cj.name}')

		animation_timer = time.time()
		check_count = 0
		new_rows = -1
		while not all(completed):
			msg = ''
			if (time.time() - animation_timer) > 0.15 or check_count == 0:
				animation_timer = time.time()
				sys_mem = psutil.virtual_memory().percent
				sys_cpu = psutil.cpu_percent()
				s = next(load_syms)

			msg += ''.join(
				cstr(
					header_str.format(level, sys_mem, sys_cpu),
					color=Colors.UNDERLINE + Colors.HEADER,
				)
			)

			# write the status of each job
			for i, cj in enumerate(jobs):
				worker = workers[i]
				memory_percent = cj.result['mem_percent']
				cpu = cj.result['cpu_percent']

				if worker.done():
					completed[i] = True
					if cj.result['returncode'] == 0:
						msg += ''.join(
							cstr(
								row_str.format(
									symbols.CHECK, cj.name, memory_percent, cpu
								),
								color=Colors.OKGREEN,
							)
						)
					else:
						msg += ''.join(
							cstr(
								row_str.format('X', cj.name, memory_percent, cpu),
								color=Colors.FAIL,
							)
						)
						failed.add(cj.name)
				elif worker.running():
					msg += row_str.format(s, cj.name, memory_percent, cpu)
			pm.clear()
			pm.cprint(msg, end='')
			time.sleep(0.01)
			check_count+=1
			sys.stdout.flush()

	pool.shutdown(wait=True)

	# if anything failed, stop the execution
	if failed:
		for j in jobs:
			if j.result['returncode'] != 0:
				header = f'{j.name} STDOUT'
				print('')
				cprint(len(header) * '~', color=Colors.WARNING)
				cprint(header, color=Colors.WARNING)
				cprint(len(header) * '~', color=Colors.WARNING)
				with open(j.logfile, 'r') as reader:
					for line in reader:
						print(line)

				header = f'{j.name} STDERR'
				print('')
				cprint(len(header) * '~', color=Colors.FAIL)
				cprint(header, color=Colors.FAIL)
				cprint(len(header) * '~', color=Colors.FAIL)
				print(str(j.result['stderr']))

		cprint('--------------------------', color=Colors.FAIL)
		raise Exception(f'Jobs {[f for f in failed]} failed.')


def validate_datapointers(
	datapointers: list[DataPointer], project_settings: ProjectSettings, level: int
):
	"""
	Validates if datapointers exist within a project.

	Parameters
	----------
	datapointers : list[DataPointer]
	    List of DataPointer objects.
	project_settings : ProjectSettings
	    Project settings
	level : int
	    Execution level in the DAG

	Raises
	------
	FileNotFoundError
	    If a dataset is missing
	"""
	# if pnts, then just check that
	# the datasets claimed to exists actually exist
	cprint(f'{level}: DataSets', color=Colors.HEADER)
	proj_dir = project_settings.project_dir
	failed = []
	for ap in datapointers:
		full = (proj_dir / ap.path).resolve()
		if full.exists():
			cprint(f'  {symbols.CHECK} | {ap.name}', color=Colors.OKGREEN)
		else:
			cprint(f'  {symbols.XBOX} | {ap.name}', color=Colors.FAIL)
			failed.append(full)
	if len(failed) > 0:
		raise FileNotFoundError(f'Expected datasets are missing: {failed}')
