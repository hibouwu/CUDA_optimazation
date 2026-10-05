"""Strict pre-IO query namespace; original reviewed ARM provider stays frozen."""
from pathlib import Path
from common.suite_io import require
from common.s16_quota_v1 import query_quota as original_query_quota

def query_destination(repo,output):
 repo=Path(repo).absolute();output=Path(output).absolute()
 require('..' not in repo.parts and '..' not in output.parts,'query path cannot contain parent traversal')
 for path in (repo,output):
  cursor=Path(path.anchor)
  for part in path.parts[1:]:
   cursor=cursor/part
   require(not cursor.is_symlink(),'query path symlink forbidden before IO')
   if cursor.exists() and cursor!=output:require(cursor.is_dir(),'query ancestor must be directory')
 if output.exists():require(output.is_file(),'query alias must be a regular file')
 require(output.is_relative_to(repo) and output!=repo and output.resolve().is_relative_to(repo.resolve()),'query output must resolve inside deployment')
 return repo.resolve(),output

def query_quota(repo,output):
 repo,output=query_destination(repo,output)
 return original_query_quota(repo,output)
