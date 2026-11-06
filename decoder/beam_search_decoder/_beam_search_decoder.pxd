#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
from libc.stdlib cimport malloc, calloc, free
from libcpp cimport bool
from libcpp.vector cimport vector
cimport numpy as np
ctypedef np.uint8_t uint8_t

cdef extern from "beam_search.hpp" namespace "ldpc::bp":
    cdef cppclass BpEntry "ldpc::bp::BpEntry":
        BpEntry() except +
        bool at_end()

    cdef cppclass BpSparse "ldpc::bp::BpSparse":
        int m
        int n
        BpSparse() except +
        BpSparse(int m, int n, int entry_count) except +
        BpEntry& insert_entry(int i, int j)
        BpEntry& get_entry(int i, int j)
        vector[uint8_t]& mulvec(vector[uint8_t]& input_vector, vector[uint8_t]& output_vector)
        vector[uint8_t] mulvec(vector[uint8_t]& input_vector)

        vector[vector[int]] nonzero_coordinates()
        int entry_count()

        int get_col_degree(int col)
        int get_row_degree(int row)

    cdef cppclass BeamSearchDecoderCpp "ldpc::bp::BeamSearchDecoder":
            BeamSearchDecoderCpp(
                BpSparse& parity_check_matrix,
                vector[double] channel_probabilities,
                int max_rounds,
                int beam_width,
                int num_results,
                int initial_iters,
                int iters_per_round,
                bool warm_start_children,
                double child_restart_alpha,
                bool child_restart_local_shells,
                double child_restart_local_shell_alpha_radius1,
                double child_restart_local_shell_alpha_radius2,
                double child_restart_local_shell_alpha_far,
                bool child_restart_adaptive_near_clamp,
                int child_restart_adaptive_flip_threshold,
                double child_restart_adaptive_flip_penalty,
                double child_restart_adaptive_disagree_penalty) except +
            BpSparse& pcm
            vector[double] channel_probabilities
            int check_count
            int bit_count
            int max_rounds
            int beam_width
            int num_results
            int initial_iters
            int iters_per_round
            bool warm_start_children
            double child_restart_alpha
            bool child_restart_local_shells
            double child_restart_local_shell_alpha_radius1
            double child_restart_local_shell_alpha_radius2
            double child_restart_local_shell_alpha_far
            bool child_restart_adaptive_near_clamp
            int child_restart_adaptive_flip_threshold
            double child_restart_adaptive_flip_penalty
            double child_restart_adaptive_disagree_penalty
            vector[uint8_t] decoding
            vector[uint8_t] candidate_syndrome
            vector[double] log_prob_ratios
            vector[double] initial_log_prob_ratios
            int iterations
            bool converge
            vector[uint8_t] decode(vector[uint8_t]& syndrome)

cdef class BeamSearchDecoderBase:
    cdef BpSparse *pcm
    cdef int m, n
    cdef vector[uint8_t] _syndrome
    cdef vector[double] _error_channel
    cdef bool MEMORY_ALLOCATED
    cdef BeamSearchDecoderCpp *bpd
    cdef str user_dtype

cdef class BeamSearchDecoder(BeamSearchDecoderBase):
    cdef vector[uint8_t] _received_vector
    pass
