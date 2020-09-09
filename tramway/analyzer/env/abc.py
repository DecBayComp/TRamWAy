
from ..attribute.abc import Attribute, abstractmethod

class Environment(Attribute):
    @property
    @abstractmethod
    def script(self):
        pass
    @abstractmethod
    def setup(self):
        pass
    @abstractmethod
    def dispatch(self, **kwargs):
        """
        Input arguments must be keyworded.

        Arguments:

            stage_index (int): index of the pipeline stage which material has to be dispatched.

            source (str): source name of the spt dataset to be dispatched.

        """
        pass
    @abstractmethod
    def make_job(self, stage_index=None, source=None, region_label=None, segment_index=None):
        pass
    @abstractmethod
    def submit_jobs(self):
        pass
    @abstractmethod
    def wait_for_job_completion(self):
        pass
    @abstractmethod
    def collect_results(self):
        pass

