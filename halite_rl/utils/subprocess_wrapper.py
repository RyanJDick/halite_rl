import atexit
import multiprocessing
import sys
import traceback


class SubProcessWrapperException(Exception):
    pass


class SubProcessWrapper():
    """A class that wraps any object and allows its methods to be called asynchronously in a
    separate process (no GIL locking).

    Not intended to be thread-safe.

    Adapted from https://github.com/google-research/batch-ppo/blob/master/agents/tools/wrappers.py.
    """

    _CALL = 1
    _RESULT = 2
    _EXCEPTION = 3
    _CLOSE = 4

    def __init__(self, constructor):
        self._parent_conn, worker_conn = multiprocessing.Pipe()
        obj = constructor()
        self._process = multiprocessing.Process(
            target=self._worker, args=(obj, worker_conn))
        self._call_in_progress = False

        atexit.register(self.close) # Close in case of unexpected termination.
        self._process.start()

    def call_sync(self, fn_name, *args, **kwargs):
        self.call_async(fn_name, *args, **kwargs)
        return self.get_result()

    def call_async(self, fn_name, *args, **kwargs):
        """Asynchronously call a method of the wrapped object.

        Parameters:
        -----------
        fn_name : str
            The name of the method to call on the wrapped object.

        *args, **kwargs
            All other args will be forwarded to the wrapped object method call.

        Returns:
        --------
        receive_fn : func
            Function to be called in the future that will block and return the result of the method
            call.

        Raises:
        -------
        SubProcessWrapperException
            Raised if an existing call is in-progress, or the wrapper has been closed.
        """
        if self._call_in_progress:
            raise SubProcessWrapperException("Existing call already in progress.")
        self._call_in_progress = True

        payload = (fn_name, args, kwargs)
        try:
            self._parent_conn.send((self._CALL, payload))
        except IOError as e:
            raise SubProcessWrapperException(f"Wrapper is closed: {str(e)}")

    def get_result(self):
        """Get result from current async call. Block until call finishes.

        Returns:
        --------
        Return value from wrapped object method.

        Raises:
        -------
        SubProcessWrapperException
            Raised if there is no asyn call in progress, or if the wrapped object method raises an
            exception.
        """
        if not self._call_in_progress:
            raise SubProcessWrapperException("Attempted to get_result, but no call in progress.")

        message, payload = self._parent_conn.recv()
        self._call_in_progress = False

        if message == self._EXCEPTION:
            raise SubProcessWrapperException(f"Exception in async call: {payload}")

        if message == self._RESULT:
            return payload

        raise SubProcessWrapperException(f"Received message of unexpected type: {message}")

        self._call_in_progress = False

    def close(self):
        """Send a close message to the worker subprocess and join it.
        """
        try:
            self._parent_conn.send((self._CLOSE, None))
            self._parent_conn.close()
        except IOError:
            # The connection was already closed.
            pass


    def _worker(self, obj, conn):
        try:
            self._worker_loop(obj, conn)
        except Exception:
            stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
            conn.send((self._EXCEPTION, stacktrace))
        conn.close()


    def _worker_loop(self, obj, conn):
        while True:
            try:
                # Only block for short periods to allow keyboard exceptions.
                if not conn.poll(0.1):
                    continue
                message, payload = conn.recv()
            except (EOFError, KeyboardInterrupt):
                break

            if message == self._CALL:
                fn_name, args, kwargs = payload
                result = getattr(obj, fn_name)(*args, **kwargs)
                conn.send((self._RESULT, result))
            elif message == self._CLOSE:
                assert payload is None
                break
            else:
                raise SubProcessWrapperException(f"Worker received unexpected message: {message}")
