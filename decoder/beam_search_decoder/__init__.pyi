import numpy as np
import scipy.sparse
from typing import Optional, List, Union
import warnings
import ldpc.helpers.scipy_helpers


def io_test(pcm: Union[scipy.sparse.spmatrix,np.ndarray]): ...



class BeamSearchDecoderBase:

    """
    BeamSearch Decoder base class
    """

    def __cinit__(self,pcm, **kwargs): ...

    def __del__(self): ...

    @property
    def error_channel(self) -> np.ndarray:
        """
        Returns the current error channel vector.

        Returns:
            np.ndarray: A numpy array containing the current error channel vector.
        """

    @error_channel.setter
    def error_channel(self, value: Union[Optional[List[float]],np.ndarray]) -> None:
        """
        Sets the error channel for the decoder.

        Args:
            value (Optional[List[float]]): The error channel vector to be set. Must have length equal to the block
            length of the code `self.n`.
        """

    def update_channel_probs(self, value: Union[List[float],np.ndarray]) -> None: ...

    @property
    def channel_probs(self) -> np.ndarray: ...

    @property
    def log_prob_ratios(self) -> np.ndarray:
        """
        Returns the current log probability ratio vector.

        Returns:
            np.ndarray: A numpy array containing the current log probability ratio vector.
        """

    @property
    def converge(self) -> bool:
        """
        Returns whether the decoder has converged or not.

        Returns:
            bool: True if the decoder has converged, False otherwise.
        """

    @property
    def iter(self) -> int:
        """
        Returns the number of iterations performed by the decoder.

        Returns:
            int: The number of iterations performed by the decoder.
        """


    @property
    def check_count(self) -> int:
        """
        Returns the number of rows of the parity check matrix.

        Returns:
            int: The number of rows of the parity check matrix.
        """

    @property
    def bit_count(self) -> int:
        """
        Returns the number of columns of the parity check matrix.

        Returns:
            int: The number of columns of the parity check matrix.
        """

    @property
    def max_rounds(self) -> int:
        """
        Returns the maximum rounds of branching allowed by the decoder.

        Returns:
            int: The maximum rounds of branching allowed by the decoder.
        """

    @max_rounds.setter
    def max_rounds(self, value: int) -> None:
        """
        Sets the maximum rounds of branching allowed by the decoder.

        Args:
            value (int): The maximum rounds of branching allowed by the decoder.

        Raises:
            ValueError: If value is not a positive integer.
        """

    @property
    def beam_width(self) -> int:
        """
        Returns the maximum list size allowed by the decoder.

        Returns:
            int: The maximum list size allowed by the decoder.
        """

    @beam_width.setter
    def beam_width(self, value: int) -> None:
        """
        Sets the maximum list size allowed by the decoder.

        Args:
            value (int): The maximum list size allowed by the decoder.

        Raises:
            ValueError: If value is not a positive integer.
        """

    @property
    def num_results(self) -> int:
        """
        Returns the number of solutions sought by the decoder.

        Returns:
            int: The number of solutions sought by the decoder.
        """

    @num_results.setter
    def num_results(self, value: int) -> None:
        """
        Sets the number of solutions sought by the decoder.

        Args:
            value (int): The number of solutions sought by the decoder.

        Raises:
            ValueError: If value is not a positive integer.
        """

    @property
    def initial_iters(self) -> int:
        """
        Returns the number of iterations in preprocessing.

        Returns:
            int: The number of iterations in preprocessing.
        """

    @initial_iters.setter
    def initial_iters(self, value: int) -> None:
        """
        Sets the number of iterations in preprocessing.

        Args:
            value (int): The number of iterations in preprocessing.

        Raises:
            ValueError: If value is not a positive integer.
        """

    @property
    def iters_per_round(self) -> int:
        """
        Returns the number of iterations in each round.

        Returns:
            int: The number of iterations in each round.
        """

    @iters_per_round.setter
    def iters_per_round(self, value: int) -> None:
        """
        Sets the number of iterations in each round.

        Args:
            value (int): The number of iterations in each round.

        Raises:
            ValueError: If value is not a positive integer.
        """

    @property
    def warm_start_children(self) -> bool:
        """
        Returns whether child paths warm-start from the parent bit-to-check messages.

        Returns:
            bool: True for warm child restarts, False for cold child restarts.
        """

    @warm_start_children.setter
    def warm_start_children(self, value) -> None:
        """
        Sets whether child paths warm-start from the parent bit-to-check messages.

        Args:
            value: Bool-like flag. True enables warm child restarts, False uses cold restarts.
        """

    @property
    def child_restart_alpha(self) -> float:
        """
        Returns the child restart mix coefficient.

        Returns:
            float: 1.0 uses the parent messages, 0.0 cold-starts from the channel prior,
            intermediate values linearly interpolate between the two.
        """

    @child_restart_alpha.setter
    def child_restart_alpha(self, value) -> None:
        """
        Sets the child restart mix coefficient.

        Args:
            value: Float in [0, 1]. 1 keeps warm restarts, 0 uses cold restarts.
        """

    @property
    def child_restart_local_shells(self) -> bool:
        """
        Returns whether child restarts use the local Tanner-shell interpolation policy.

        Returns:
            bool: True when shell-local interpolation is enabled.
        """

    @child_restart_local_shells.setter
    def child_restart_local_shells(self, value) -> None:
        """
        Enables or disables shell-local child restart interpolation.

        Args:
            value: Bool-like flag.
        """

    @property
    def child_restart_local_shell_alpha_radius1(self) -> float: ...

    @child_restart_local_shell_alpha_radius1.setter
    def child_restart_local_shell_alpha_radius1(self, value) -> None: ...

    @property
    def child_restart_local_shell_alpha_radius2(self) -> float: ...

    @child_restart_local_shell_alpha_radius2.setter
    def child_restart_local_shell_alpha_radius2(self, value) -> None: ...

    @property
    def child_restart_local_shell_alpha_far(self) -> float: ...

    @child_restart_local_shell_alpha_far.setter
    def child_restart_local_shell_alpha_far(self, value) -> None: ...

    @property
    def child_restart_adaptive_near_clamp(self) -> bool: ...

    @child_restart_adaptive_near_clamp.setter
    def child_restart_adaptive_near_clamp(self, value) -> None: ...

    @property
    def child_restart_adaptive_flip_threshold(self) -> int: ...

    @child_restart_adaptive_flip_threshold.setter
    def child_restart_adaptive_flip_threshold(self, value) -> None: ...

    @property
    def child_restart_adaptive_flip_penalty(self) -> float: ...

    @child_restart_adaptive_flip_penalty.setter
    def child_restart_adaptive_flip_penalty(self, value) -> None: ...

    @property
    def child_restart_adaptive_disagree_penalty(self) -> float: ...

    @child_restart_adaptive_disagree_penalty.setter
    def child_restart_adaptive_disagree_penalty(self, value) -> None: ...


class BeamSearchDecoder(BeamSearchDecoderBase):
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
                 **kwargs): ...

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
                 **kwargs): ...

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


    @property
    def decoding(self) -> np.ndarray:
        """
        Returns the current decoded output.

        Returns:
            np.ndarray: A numpy array containing the current decoded output.
        """
