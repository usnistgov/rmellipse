__all__ = ["UninitializedRMEProject", "JobFailure"]


class UninitializedRMEProject(Exception):
    def __init__(*args, **kwargs):
        Exception.__init__(*args, **kwargs)


class JobFailure(Exception):
    def __init__(*args, **kwargs):
        Exception.__init__(*args, **kwargs)
