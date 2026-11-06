#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
import numpy as np
import scipy.sparse
from typing import Optional, List, Union
import warnings
import ldpc.helpers.scipy_helpers

cdef BpSparse* Py2BpSparse(pcm):

    cdef int m
    cdef int n
    cdef int nonzero_count

    #check the parity check matrix is the right type
    if isinstance(pcm, np.ndarray) or isinstance(pcm, scipy.sparse.spmatrix):
        pass
    else:
        raise TypeError(f"The input matrix is of an invalid type. Please input\
        a np.ndarray or scipy.sparse.spmatrix object, not {type(pcm)}")

    # Convert to binary sparse matrix and validate input
    pcm = ldpc.helpers.scipy_helpers.convert_to_binary_sparse(pcm)

    # get the parity check dimensions
    m, n = pcm.shape[0], pcm.shape[1]


    # get the number of nonzero entries in the parity check matrix
    if isinstance(pcm,np.ndarray):
        nonzero_count  = int(np.sum( np.count_nonzero(pcm,axis=1) ))
    elif isinstance(pcm,scipy.sparse.spmatrix):
        nonzero_count = int(pcm.nnz)

    # Matrix memory allocation
    cdef BpSparse* cpcm = new BpSparse(m,n,nonzero_count) #creates the C++ sparse matrix object

    #fill sparse matrix
    if isinstance(pcm,np.ndarray):
        for i in range(m):
            for j in range(n):
                if pcm[i,j]==1:
                    cpcm.insert_entry(i,j)
    elif isinstance(pcm,scipy.sparse.spmatrix):
        rows, cols = pcm.nonzero()
        for i in range(len(rows)):
            cpcm.insert_entry(rows[i], cols[i])

    return cpcm

cdef coords_to_scipy_sparse(vector[vector[int]]& entries, int m, int n, int entry_count):

    cdef np.ndarray[int, ndim=1] rows = np.zeros(entry_count, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] cols = np.zeros(entry_count, dtype=np.int32)
    cdef np.ndarray[uint8_t, ndim=1] data = np.ones(entry_count, dtype=np.uint8)

    for i in range(entry_count):
        rows[i] = entries[i][0]
        cols[i] = entries[i][1]

    smat = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(m, n), dtype=np.uint8)
    return smat

cdef BpSparse2Py(BpSparse* cpcm):
    cdef int i
    cdef int m = cpcm.m
    cdef int n = cpcm.n
    cdef int entry_count = cpcm.entry_count()
    cdef vector[vector[int]] entries = cpcm.nonzero_coordinates()
    smat = coords_to_scipy_sparse(entries, m, n, entry_count)
    return smat


def io_test(pcm: Union[scipy.sparse.spmatrix,np.ndarray]):
    cdef BpSparse* cpcm = Py2BpSparse(pcm)
    output = BpSparse2Py(cpcm)
    del cpcm
    return output



cdef class BeamSearchDecoderBase:

    """
    BeamSearch Decoder base class
    """

    def __cinit__(self,pcm, **kwargs):

        error_channel=kwargs.get("error_channel", None)
        max_rounds=kwargs.get("max_rounds",10)
        beam_width=kwargs.get("beam_width",8)
        num_results=kwargs.get("num_results",1)
        initial_iters=kwargs.get("initial_iters",30)
        iters_per_round=kwargs.get("iters_per_round",20)
        warm_start_children=kwargs.get("warm_start_children", True)
        child_restart_alpha=kwargs.get("child_restart_alpha", None)
        child_restart_local_shells=kwargs.get("child_restart_local_shells", False)
        child_restart_local_shell_alpha_radius1=kwargs.get("child_restart_local_shell_alpha_radius1", 0.0)
        child_restart_local_shell_alpha_radius2=kwargs.get("child_restart_local_shell_alpha_radius2", 0.5)
        child_restart_local_shell_alpha_far=kwargs.get("child_restart_local_shell_alpha_far", 1.0)
        child_restart_adaptive_near_clamp=kwargs.get("child_restart_adaptive_near_clamp", False)
        child_restart_adaptive_flip_threshold=kwargs.get("child_restart_adaptive_flip_threshold", 2)
        child_restart_adaptive_flip_penalty=kwargs.get("child_restart_adaptive_flip_penalty", 0.25)
        child_restart_adaptive_disagree_penalty=kwargs.get("child_restart_adaptive_disagree_penalty", 0.5)
        channel_probs = kwargs.get("channel_probs", [None])

        """
        Docstring test
        """

        cdef int i, j, nonzero_count
        self.MEMORY_ALLOCATED=False

        # Matrix memory allocation
        if isinstance(pcm, np.ndarray) or isinstance(pcm, scipy.sparse.spmatrix):
            pass
        else:
            raise TypeError(f"The input matrix is of an invalid type. Please input\
            a np.ndarray or scipy.sparse.spmatrix object, not {type(pcm)}")
        self.pcm = Py2BpSparse(pcm)

        # get the parity check dimensions
        self.m, self.n = pcm.shape[0], pcm.shape[1]

        # allocate vectors for decoder input
        self._error_channel.resize(self.n) #C++ vector for the error channel
        self._syndrome.resize(self.m) #C++ vector for the syndrome



        ## initialise the decoder with default values
        self.bpd = new BeamSearchDecoderCpp(self.pcm[0],self._error_channel,10,8,1,30,20,True,1.0,False,0.0,0.5,1.0,False,2,0.25,0.5)

        ## set the decoder parameters
        self.max_rounds = max_rounds
        self.beam_width = beam_width
        self.num_results = num_results
        self.initial_iters = initial_iters
        self.iters_per_round = iters_per_round
        if child_restart_alpha is None:
            self.warm_start_children = warm_start_children
        else:
            self.child_restart_alpha = child_restart_alpha
        self.child_restart_local_shells = child_restart_local_shells
        self.child_restart_local_shell_alpha_radius1 = child_restart_local_shell_alpha_radius1
        self.child_restart_local_shell_alpha_radius2 = child_restart_local_shell_alpha_radius2
        self.child_restart_local_shell_alpha_far = child_restart_local_shell_alpha_far
        self.child_restart_adaptive_near_clamp = child_restart_adaptive_near_clamp
        self.child_restart_adaptive_flip_threshold = child_restart_adaptive_flip_threshold
        self.child_restart_adaptive_flip_penalty = child_restart_adaptive_flip_penalty
        self.child_restart_adaptive_disagree_penalty = child_restart_adaptive_disagree_penalty

        if error_channel is not None:
            self.error_channel = error_channel
        else:
            raise ValueError("Please specify the error channel. error_channel:\
            list of floats of length equal to the block length of the code {self.n}.")




        self.MEMORY_ALLOCATED=True

    def __del__(self):
        if self.MEMORY_ALLOCATED:
            del self.bpd
            del self.pcm

    @property
    def error_channel(self) -> np.ndarray:
        """
        Returns the current error channel vector.

        Returns:
            np.ndarray: A numpy array containing the current error channel vector.
        """
        out = np.zeros(self.n).astype(float)
        for i in range(self.n):
            out[i] = self.bpd.channel_probabilities[i]
        return out

    @error_channel.setter
    def error_channel(self, value: Union[Optional[List[float]],np.ndarray]) -> None:
        """
        Sets the error channel for the decoder.

        Args:
            value (Optional[List[float]]): The error channel vector to be set. Must have length equal to the block
            length of the code `self.n`.
        """
        if value is not None:
            if len(value) != self.n:
                raise ValueError(f"The error channel vector must have length {self.n}, not {len(value)}.")
            for i in range(self.n):
                self.bpd.channel_probabilities[i] = value[i]

    def update_channel_probs(self, value: Union[List[float],np.ndarray]) -> None:
        self.error_channel = value

    @property
    def channel_probs(self) -> np.ndarray:
        out = np.zeros(self.n).astype(float)
        for i in range(self.n):
            out[i] = self.bpd.channel_probabilities[i]
        return out

    @property
    def log_prob_ratios(self) -> np.ndarray:
        """
        Returns the current log probability ratio vector.

        Returns:
            np.ndarray: A numpy array containing the current log probability ratio vector.
        """
        out = np.zeros(self.n)
        for i in range(self.n):
            out[i] = self.bpd.log_prob_ratios[i]
        return out

    @property
    def converge(self) -> bool:
        """
        Returns whether the decoder has converged or not.

        Returns:
            bool: True if the decoder has converged, False otherwise.
        """
        return self.bpd.converge

    @property
    def iter(self) -> int:
        """
        Returns the number of iterations performed by the decoder.

        Returns:
            int: The number of iterations performed by the decoder.
        """
        return self.bpd.iterations


    @property
    def check_count(self) -> int:
        """
        Returns the number of rows of the parity check matrix.

        Returns:
            int: The number of rows of the parity check matrix.
        """
        return self.bpd.pcm.m

    @property
    def bit_count(self) -> int:
        """
        Returns the number of columns of the parity check matrix.

        Returns:
            int: The number of columns of the parity check matrix.
        """
        return self.bpd.pcm.n

    @property
    def max_rounds(self) -> int:
        """
        Returns the maximum rounds of branching allowed by the decoder.

        Returns:
            int: The maximum rounds of branching allowed by the decoder.
        """
        return self.bpd.max_rounds

    @max_rounds.setter
    def max_rounds(self, value: int) -> None:
        """
        Sets the maximum rounds of branching allowed by the decoder.

        Args:
            value (int): The maximum rounds of branching allowed by the decoder.

        Raises:
            ValueError: If value is not a positive integer.
        """
        if not isinstance(value, int):
            raise ValueError("max_rounds input parameter is invalid. This must be specified as a positive int.")
        if value < 0:
            raise ValueError(f"max_rounds input parameter must be a positive int. Not {value}.")
        self.bpd.max_rounds = value if value != 0 else 1

    @property
    def beam_width(self) -> int:
        """
        Returns the maximum list size allowed by the decoder.

        Returns:
            int: The maximum list size allowed by the decoder.
        """
        return self.bpd.beam_width

    @beam_width.setter
    def beam_width(self, value: int) -> None:
        """
        Sets the maximum list size allowed by the decoder.

        Args:
            value (int): The maximum list size allowed by the decoder.

        Raises:
            ValueError: If value is not a positive integer.
        """
        if not isinstance(value, int):
            raise ValueError("beam_width input parameter is invalid. This must be specified as a positive int.")
        if value < 0:
            raise ValueError(f"beam_width input parameter must be a positive int. Not {value}.")
        self.bpd.beam_width = value if value != 0 else 8

    @property
    def num_results(self) -> int:
        """
        Returns the number of solutions sought by the decoder.

        Returns:
            int: The number of solutions sought by the decoder.
        """
        return self.bpd.num_results

    @num_results.setter
    def num_results(self, value: int) -> None:
        """
        Sets the number of solutions sought by the decoder.

        Args:
            value (int): The number of solutions sought by the decoder.

        Raises:
            ValueError: If value is not a positive integer.
        """
        if not isinstance(value, int):
            raise ValueError("num_results input parameter is invalid. This must be specified as a positive int.")
        if value < 0:
            raise ValueError(f"num_results input parameter must be a positive int. Not {value}.")
        self.bpd.num_results = value if value != 0 else 5

    @property
    def initial_iters(self) -> int:
        """
        Returns the number of iterations in preprocessing.

        Returns:
            int: The number of iterations in preprocessing.
        """
        return self.bpd.initial_iters

    @initial_iters.setter
    def initial_iters(self, value: int) -> None:
        """
        Sets the number of iterations in preprocessing.

        Args:
            value (int): The number of iterations in preprocessing.

        Raises:
            ValueError: If value is not a positive integer.
        """
        if not isinstance(value, int):
            raise ValueError("initial_iters input parameter is invalid. This must be specified as a positive int.")
        if value < 0:
            raise ValueError(f"initial_iters input parameter must be a positive int. Not {value}.")
        self.bpd.initial_iters = value

    @property
    def iters_per_round(self) -> int:
        """
        Returns the number of iterations in each round.

        Returns:
            int: The number of iterations in each round.
        """
        return self.bpd.iters_per_round

    @iters_per_round.setter
    def iters_per_round(self, value: int) -> None:
        """
        Sets the number of iterations in each round.

        Args:
            value (int): The number of iterations in each round.

        Raises:
            ValueError: If value is not a positive integer.
        """
        if not isinstance(value, int):
            raise ValueError("iters_per_round input parameter is invalid. This must be specified as a positive int.")
        if value < 0:
            raise ValueError(f"iters_per_round input parameter must be a positive int. Not {value}.")
        self.bpd.iters_per_round = value

    @property
    def warm_start_children(self) -> bool:
        """
        Returns whether child paths warm-start from the parent bit-to-check messages.

        Returns:
            bool: True for warm child restarts, False for cold child restarts.
        """
        return self.bpd.warm_start_children

    @warm_start_children.setter
    def warm_start_children(self, value) -> None:
        """
        Sets whether child paths warm-start from the parent bit-to-check messages.

        Args:
            value: Bool-like flag. True enables warm child restarts, False uses cold restarts.
        """
        self.bpd.warm_start_children = True if value else False
        self.bpd.child_restart_alpha = 1.0 if value else 0.0

    @property
    def child_restart_alpha(self) -> float:
        """
        Returns the child restart mix coefficient.

        Returns:
            float: 1.0 uses the parent messages, 0.0 cold-starts from the channel prior,
            intermediate values linearly interpolate between the two.
        """
        return self.bpd.child_restart_alpha

    @child_restart_alpha.setter
    def child_restart_alpha(self, value) -> None:
        """
        Sets the child restart mix coefficient.

        Args:
            value: Float in [0, 1]. 1 keeps warm restarts, 0 uses cold restarts.
        """
        cdef double alpha
        alpha = float(value)
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"child_restart_alpha must be in [0,1]. Not {value}.")
        self.bpd.child_restart_alpha = alpha
        self.bpd.warm_start_children = True if alpha > 0.0 else False

    @property
    def child_restart_local_shells(self) -> bool:
        """
        Returns whether child restarts use the local Tanner-shell interpolation policy.

        Returns:
            bool: True when shell-local interpolation is enabled.
        """
        return self.bpd.child_restart_local_shells

    @child_restart_local_shells.setter
    def child_restart_local_shells(self, value) -> None:
        """
        Enables or disables shell-local child restart interpolation.

        Args:
            value: Bool-like flag.
        """
        self.bpd.child_restart_local_shells = True if value else False

    @property
    def child_restart_local_shell_alpha_radius1(self) -> float:
        return self.bpd.child_restart_local_shell_alpha_radius1

    @child_restart_local_shell_alpha_radius1.setter
    def child_restart_local_shell_alpha_radius1(self, value) -> None:
        cdef double alpha
        alpha = float(value)
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"child_restart_local_shell_alpha_radius1 must be in [0,1]. Not {value}.")
        self.bpd.child_restart_local_shell_alpha_radius1 = alpha

    @property
    def child_restart_local_shell_alpha_radius2(self) -> float:
        return self.bpd.child_restart_local_shell_alpha_radius2

    @child_restart_local_shell_alpha_radius2.setter
    def child_restart_local_shell_alpha_radius2(self, value) -> None:
        cdef double alpha
        alpha = float(value)
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"child_restart_local_shell_alpha_radius2 must be in [0,1]. Not {value}.")
        self.bpd.child_restart_local_shell_alpha_radius2 = alpha

    @property
    def child_restart_local_shell_alpha_far(self) -> float:
        return self.bpd.child_restart_local_shell_alpha_far

    @child_restart_local_shell_alpha_far.setter
    def child_restart_local_shell_alpha_far(self, value) -> None:
        cdef double alpha
        alpha = float(value)
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"child_restart_local_shell_alpha_far must be in [0,1]. Not {value}.")
        self.bpd.child_restart_local_shell_alpha_far = alpha

    @property
    def child_restart_adaptive_near_clamp(self) -> bool:
        return self.bpd.child_restart_adaptive_near_clamp

    @child_restart_adaptive_near_clamp.setter
    def child_restart_adaptive_near_clamp(self, value) -> None:
        self.bpd.child_restart_adaptive_near_clamp = True if value else False

    @property
    def child_restart_adaptive_flip_threshold(self) -> int:
        return self.bpd.child_restart_adaptive_flip_threshold

    @child_restart_adaptive_flip_threshold.setter
    def child_restart_adaptive_flip_threshold(self, value) -> None:
        cdef int threshold
        threshold = int(value)
        if threshold < 1:
            raise ValueError(f"child_restart_adaptive_flip_threshold must be >=1. Not {value}.")
        self.bpd.child_restart_adaptive_flip_threshold = threshold

    @property
    def child_restart_adaptive_flip_penalty(self) -> float:
        return self.bpd.child_restart_adaptive_flip_penalty

    @child_restart_adaptive_flip_penalty.setter
    def child_restart_adaptive_flip_penalty(self, value) -> None:
        cdef double penalty
        penalty = float(value)
        if penalty < 0.0 or penalty > 1.0:
            raise ValueError(f"child_restart_adaptive_flip_penalty must be in [0,1]. Not {value}.")
        self.bpd.child_restart_adaptive_flip_penalty = penalty

    @property
    def child_restart_adaptive_disagree_penalty(self) -> float:
        return self.bpd.child_restart_adaptive_disagree_penalty

    @child_restart_adaptive_disagree_penalty.setter
    def child_restart_adaptive_disagree_penalty(self, value) -> None:
        cdef double penalty
        penalty = float(value)
        if penalty < 0.0 or penalty > 1.0:
            raise ValueError(f"child_restart_adaptive_disagree_penalty must be in [0,1]. Not {value}.")
        self.bpd.child_restart_adaptive_disagree_penalty = penalty


cdef class BeamSearchDecoder(BeamSearchDecoderBase):
    """
    Belief propagation decoder for binary linear codes.

    This class provides an implementation of belief propagation decoding for binary linear codes. The decoder uses a sparse
    parity check matrix to decode received codewords. The decoding algorithm can be configured using various parameters,
    such as the belief propagation method used, the scheduling method used, and the maximum number of iterations.

    Parameters
    ----------
    pcm : Union[np.ndarray, scipy.sparse.spmatrix]
        The parity check matrix of the binary linear code, represented as a NumPy array or a SciPy sparse matrix.
    error_channel : Optional[List[float]], optional
        The initial error channel probabilities for the decoder, by default None.
    """

    def __cinit__(self, pcm: Union[np.ndarray, scipy.sparse.spmatrix],
                 error_channel: Optional[Union[np.ndarray,List[float]]] = None, max_rounds: Optional[int] = 10,
                 beam_width: Optional[int] = 8, num_results: Optional[int] = 1, initial_iters: Optional[int] = 30,
                 iters_per_round: Optional[int] = 20, warm_start_children: Optional[bool] = True,
                 child_restart_alpha: Optional[float] = None,
                 child_restart_local_shells: Optional[bool] = False,
                 child_restart_local_shell_alpha_radius1: Optional[float] = 0.0,
                 child_restart_local_shell_alpha_radius2: Optional[float] = 0.5,
                 child_restart_local_shell_alpha_far: Optional[float] = 1.0,
                 child_restart_adaptive_near_clamp: Optional[bool] = False,
                 child_restart_adaptive_flip_threshold: Optional[int] = 2,
                 child_restart_adaptive_flip_penalty: Optional[float] = 0.25,
                 child_restart_adaptive_disagree_penalty: Optional[float] = 0.5,
                 **kwargs):

        for key in kwargs.keys():
            if key not in ["channel_probs"]:
                raise ValueError(f"Unknown parameter '{key}' passed to the BeamSearchDecoder constructor.")

        pass

    def __init__(self, pcm: Union[np.ndarray, scipy.sparse.spmatrix],
                 error_channel: Optional[Union[np.ndarray,List[float]]] = None, max_rounds: Optional[int] = 10,
                 beam_width: Optional[int] = 8, num_results: Optional[int] = 1, initial_iters: Optional[int] = 30,
                 iters_per_round: Optional[int] = 20, warm_start_children: Optional[bool] = True,
                 child_restart_alpha: Optional[float] = None,
                 child_restart_local_shells: Optional[bool] = False,
                 child_restart_local_shell_alpha_radius1: Optional[float] = 0.0,
                 child_restart_local_shell_alpha_radius2: Optional[float] = 0.5,
                 child_restart_local_shell_alpha_far: Optional[float] = 1.0,
                 child_restart_adaptive_near_clamp: Optional[bool] = False,
                 child_restart_adaptive_flip_threshold: Optional[int] = 2,
                 child_restart_adaptive_flip_penalty: Optional[float] = 0.25,
                 child_restart_adaptive_disagree_penalty: Optional[float] = 0.5,
                 **kwargs):

        if child_restart_alpha is None:
            self.warm_start_children = warm_start_children
        else:
            self.child_restart_alpha = child_restart_alpha
        self.child_restart_local_shells = child_restart_local_shells
        self.child_restart_local_shell_alpha_radius1 = child_restart_local_shell_alpha_radius1
        self.child_restart_local_shell_alpha_radius2 = child_restart_local_shell_alpha_radius2
        self.child_restart_local_shell_alpha_far = child_restart_local_shell_alpha_far
        self.child_restart_adaptive_near_clamp = child_restart_adaptive_near_clamp
        self.child_restart_adaptive_flip_threshold = child_restart_adaptive_flip_threshold
        self.child_restart_adaptive_flip_penalty = child_restart_adaptive_flip_penalty
        self.child_restart_adaptive_disagree_penalty = child_restart_adaptive_disagree_penalty

    def decode(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Decode the input input_vector using belief propagation decoding algorithm.

        Parameters
        ----------
        input_vector : numpy.ndarray
            A 1D numpy array of length equal to the number of rows in the parity check matrix.

        Returns
        -------
        numpy.ndarray
            A 1D numpy array of length equal to the number of columns in the parity check matrix.

        Raises
        ------
        ValueError
            If the length of the input input_vector does not match the number of rows in the parity check matrix.
        """

        cdef int i
        cdef bool zero_input_vector = True
        DTYPE = input_vector.dtype

        cdef int len_input_vector = len(input_vector)

        for i in range(len_input_vector):
            self._syndrome[i] = input_vector[i]
            if self._syndrome[i]: zero_input_vector = False
        if zero_input_vector:
            self.bpd.converge = True
            return np.zeros(self.bit_count,dtype=DTYPE)
        self.bpd.decode(self._syndrome)

        out = np.zeros(self.n,dtype=DTYPE)
        for i in range(self.n): out[i] = self.bpd.decoding[i]
        return out


    @property
    def decoding(self) -> np.ndarray:
        """
        Returns the current decoded output.

        Returns:
            np.ndarray: A numpy array containing the current decoded output.
        """
        out = np.zeros(self.n).astype(int)
        for i in range(self.n):
            out[i] = self.bpd.decoding[i]
        return out
