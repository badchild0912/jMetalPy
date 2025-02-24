from abc import abstractmethod, ABCMeta

"""
.. module:: Observable
   :platform: Unix, Windows
   :synopsis: Implementation of the observer-observable pattern.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class Observer():
    __metaclass__ = ABCMeta

    @abstractmethod
    def update(self, *args, **kwargs):
        """ Update method.
        """
        pass


class Observable():
    __metaclass__ = ABCMeta

    @abstractmethod
    def register(self, observer):
        pass

    @abstractmethod
    def deregister(self, observer):
        pass

    @abstractmethod
    def deregister_all(self):
        pass

    @abstractmethod
    def notify_all(self, *args, **kwargs):
        pass


