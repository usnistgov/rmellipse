"""
This class contains helpers for interacting with a CDCS instance.

In general, this module provides a set of functions for utilizing
the CachedCurator object, which is an extension of the cdcs.CDCS
that provides some extended functionality to support the use case
of this package.

Functions for the rmellipse CLI should used the functions provided by
this module where-possible, to maintain a readable, functional flow
to the command line tooling.

Returns
-------
_type_
    _description_

Raises
------
Exception
    _description_
"""

import cdcs
import json
import hashlib
from pathlib import Path
import io
import requests
import os


class HashNotInDatabaseError(Exception):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)


class AuthenticationError(Exception):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)


class ObjectNotFoundError(Exception):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)


class InternalServerError(Exception):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)


class BadResponseError(Exception):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)


status_codes = {
	200: None,
	403: AuthenticationError,
	404: ObjectNotFoundError,
	500: InternalServerError,
}


def raise_from_status_code(response):
	code = response.status_code
	try:
		error = status_codes[code]
		if error is None:
			return
	except KeyError:
		error - BadResponseError
	msg = f'- error: failed: {response.status_code} - {response.text}'
	raise error(msg)


class CachedCurator(
	cdcs.CDCS,
):
	"""
	Extension of the CDCS curator that caches information locally.

	Provides pseudo-singleton behaviour, where curator's with connection
	to a host-url are cached as a class attribute. Attempting to
	connect to the same URL twice returns the same connection. This
	is so functions operating on the same CDCS host can be written
	to run individually, then chained together arbitrarily without
	making repeated connections to the host, and making the same
	queries over HTTPS redundantly.
	"""

	curator_cache = {}

	def __new__(cls, url, *args, **kwargs):
		if url not in cls.curator_cache:
			cls.curator_cache[url] = super().__new__(cls)
		return cls.curator_cache[url]

	def __init__(self, url, *args, **kwargs):
		super().__init__(url, *args, **kwargs)
		"""For caching template dataframes"""
		self._template_df_cache = {}

		"""For caching workspace dataframes"""
		self._workspace_df_cache = {}

		"""keys are SHA1 hashes, values are PIDs."""
		self._SHA1_cache = {}

	def cached_workspace_id(
		self,
		title: str,
	) -> str:
		# cache the workspace query
		# currently returns the most recent thing
		if title not in self._workspace_df_cache:
			wrkspace_df = self.get_workspace(title)
			self._workspace_df_cache[title] = wrkspace_df
		# access the cache and return the id
		wrkspace_df = self._workspace_df_cache[title]
		# it's possible for this to return a series instead
		# of a data frame, in which case this will fail
		try:
			return str(wrkspace_df.sort_values(by='id').iloc[-1].id)
		except TypeError:
			return  str(wrkspace_df.id)

	def cached_template_id(self, title: str) -> str:
		"""
		Return and cache the template_id for a given template_name.

		Retrieves the most recent ID.

		Parameters
		----------
		title : str
		    Title of the template you are looking for

		Returns
		-------
		str
		    template_id
		"""
		# cache the template ID info if not already
		# present
		if title not in self._template_df_cache:
			template_df = self.templates_dataframe(template=title)
			self._template_df_cache[title] = template_df

		# get the most recent and return as a string
		template_df = self._template_df_cache[title]
		tid = str(template_df.sort_values(by='id').iloc[-1].id)
		return tid

	def add_sha1_pid(self, sha1: str, pid: str):
		self._SHA1_cache[sha1] = pid

	def get_sha1_pid(self, sha1: str, verbose: bool = True):
		# check if sha1 is in local cache
		# query CDCS for the , verbose = True
		# if not available, raise a unique
		# exception that can be handled,
		# so the file can be uploaded then
		# the sha1 can be cached for later
		try:
			return self._SHA1_cache[sha1]
		except KeyError:
			#  try to query the database
			result = self.query(
				template='BlobMetadata',
				parse_dates=False,
				mongoquery={'sha1': sha1},
				current=False,
				progress_bar=False,
			)
			# keep most recent id, if there
			# are multiple then that means there
			# are redundant uploads
			try:
				newest = result.sort_values(by='id').iloc[-1]
			except IndexError as e:
				raise HashNotInDatabaseError from e

			content = json.loads(newest.content)
			blob_pid = content['blob/PID/']
			self._SHA1_cache[sha1] = blob_pid
			return blob_pid


def login(
	hostname='http://127.0.0.1', username='', password='', verbose=True
) -> CachedCurator:
	if verbose:
		print('status: logging in to cdcs ...')
	host_url = hostname
	c = CachedCurator(host_url, username=username, password=password, verify=False)
	if verbose:
		version = '.'.join([str(ci) for ci in c.cdcsversion])
		print(f'{host_url} @ {version}')
	return c


def upload_record(
	*,
	curator: CachedCurator,
	title: str,
	template_title: str,
	content: dict,
	workspace_title: str = None,
	verbose: bool = False,
):
	# print(f'- status: uploading jrec {title}')
	response = None
	record_id = 1

	template_id = curator.cached_template_id(template_title)
	data = {
		'title': title,
		'template': template_id,
		'content': json.dumps(content),
	}
	# if a workspace was specified, upload there
	# not specifed defaults to the private workspace
	if workspace_title is not None:
		workspace_id = curator.cached_workspace_id(workspace_title)
		data.update({'workspace': workspace_id})

	rest_url = '/rest/data/'
	response = curator.post(rest_url, json=data)

	if response.status_code == 201:
		response_data = response.json()
		if verbose:
			print(f'- status: success: json record {title} created')
		record_id = response_data.get('id')
	else:
		if verbose:
			print(f'- error: failed: {response.status_code} - {response.text}')

	return response


def upload_json_schema(curator: CachedCurator, title: str, filename: str, content={}):
	print(f'- status: uploading jschmema {title}')
	response = None
	template_id = 1
	data = {'title': title, 'filename': filename, 'content': json.dumps(content)}
	rest_url = '/rest/template/global/'
	response = curator.post(rest_url, json=data)

	if response.status_code == 201:
		response_data = response.json()
		print(f'- status: success: json schema {title} created')
		template_id = response_data.get('id')
	else:
		raise Exception(f'- error: failed: {response.status_code} - {response.text}')

	return response, template_id


def stream_blob_to_file(
	curator: CachedCurator,
	blob_pid: str,
	target_file: Path,
	update_dict: dict,
	expected_size: int,
	chunk_size=2**25,
):
	"""
	Stream a blob on CDCS to a target file.

	Parameters
	----------
	blob_pid : str
		PID (i.e. unique url) of the blob.
	target_file : Path
		Target file path.
	"""
	# with open(target_file, 'wb') as out_file:
	# 	content = requests.get(blob_pid, stream=True).content
	# 	out_file.write(content)
	try:
		url_str = blob_pid.split(curator.host)[-1]
		with curator.get(url_str, stream=True) as r:
			r.raise_for_status()
			with open(target_file, 'wb') as f:
				for chunk in r.iter_content(chunk_size=chunk_size):
					# If you have chunk encoded response uncomment if
					# and set chunk_size parameter to None.
					# if chunk:
					update_dict['size'] += chunk_size
					if update_dict['size'] > expected_size:
						update_dict['size'] = expected_size
					f.write(chunk)
		update_dict['success'] = True
	except Exception as e:
		update_dict['error'] = str(e)
		update_dict['success'] = False
	update_dict['finished'] = True
