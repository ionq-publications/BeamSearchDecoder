// This file is part of BeamSearchDecoder.

// Copyright (c) 2025 IonQ, Inc., all rights reserved

// Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
// 4.0 International License (CC BY-NC-SA 4.0).

// You may obtain a copy of the License at:
// https://creativecommons.org/licenses/by-nc-sa/4.0/


#ifndef BP_H
#define BP_H

#include <utility>
#include <algorithm>
#include <iostream>
#include <vector>
#include <queue>
#include <unordered_set>
#include <memory>
#include <iterator>
#include <cmath>
#include <limits>
#include <random>
#include <chrono>
#include <stdexcept> // required for std::runtime_error
#include <set>
#include <cstdint>

#include "math.h"
#include "sparse_matrix_base.hpp"
#include "gf2sparse.hpp"

namespace ldpc {
    namespace bp {
        class BpEntry : public ldpc::sparse_matrix_base::EntryBase<BpEntry> {
        public:
            double bit_to_check_msg = 0.0;
            double check_to_bit_msg = 0.0;

            ~BpEntry() = default;
        };
        using BpSparse = ldpc::gf2sparse::GF2Sparse<BpEntry>;

        class BeamSearchDecoder {
            // TODO properties should be private and only accessible via getters and setters
        public:
            BpSparse &pcm;
            std::vector<double> channel_probabilities;
            int check_count;
            int bit_count;
            int max_rounds;
            int beam_width;
            int num_results;
            int initial_iters;
            int iters_per_round;
            bool warm_start_children;
            double child_restart_alpha;
            bool child_restart_local_shells;
            double child_restart_local_shell_alpha_radius1;
            double child_restart_local_shell_alpha_radius2;
            double child_restart_local_shell_alpha_far;
            bool child_restart_adaptive_near_clamp;
            int child_restart_adaptive_flip_threshold;
            double child_restart_adaptive_flip_penalty;
            double child_restart_adaptive_disagree_penalty;
            std::vector<uint8_t> decoding;
            std::vector<uint8_t> candidate_syndrome;
            std::vector<std::vector<uint8_t>> child_restart_adjacent_checks_by_bit;
            std::vector<std::vector<uint8_t>> child_restart_shell2_bits_by_bit;

            std::vector<double> log_prob_ratios;
            std::vector<double> initial_log_prob_ratios;
            int iterations;
            bool converge;
            uint64_t decode_calls;

            BeamSearchDecoder(
                    BpSparse &parity_check_matrix,
                    std::vector<double> channel_probabilities,
                    int max_rounds = 10,
                    int beam_width = 8,
                    int num_results = 1,
                    int initial_iters = 30,
                    int iters_per_round = 20,
                    bool warm_start_children = true,
                    double child_restart_alpha = 1.0,
                    bool child_restart_local_shells = false,
                    double child_restart_local_shell_alpha_radius1 = 0.0,
                    double child_restart_local_shell_alpha_radius2 = 0.5,
                    double child_restart_local_shell_alpha_far = 1.0,
                    bool child_restart_adaptive_near_clamp = false,
                    int child_restart_adaptive_flip_threshold = 2,
                    double child_restart_adaptive_flip_penalty = 0.25,
                    double child_restart_adaptive_disagree_penalty = 0.5) :
                    pcm(parity_check_matrix), channel_probabilities(std::move(channel_probabilities)),
                    check_count(pcm.m), bit_count(pcm.n), max_rounds(max_rounds), beam_width(beam_width), num_results(num_results),
                    initial_iters(initial_iters), iters_per_round(iters_per_round), warm_start_children(warm_start_children),
                    child_restart_alpha(child_restart_alpha), child_restart_local_shells(child_restart_local_shells),
                    child_restart_local_shell_alpha_radius1(child_restart_local_shell_alpha_radius1),
                    child_restart_local_shell_alpha_radius2(child_restart_local_shell_alpha_radius2),
                    child_restart_local_shell_alpha_far(child_restart_local_shell_alpha_far),
                    child_restart_adaptive_near_clamp(child_restart_adaptive_near_clamp),
                    child_restart_adaptive_flip_threshold(child_restart_adaptive_flip_threshold),
                    child_restart_adaptive_flip_penalty(child_restart_adaptive_flip_penalty),
                    child_restart_adaptive_disagree_penalty(child_restart_adaptive_disagree_penalty),
                    iterations(0), decode_calls(0) //the parity check matrix is passed in by reference
            {

                this->initial_log_prob_ratios.resize(bit_count);
                this->log_prob_ratios.resize(bit_count);
                this->candidate_syndrome.resize(check_count);
                this->decoding.resize(bit_count);
                this->converge = 0;


                if (this->channel_probabilities.size() != this->bit_count) {
                    throw std::runtime_error(
                            "Channel probabilities vector must have length equal to the number of bits");
                }

                this->prepare_child_restart_shell_metadata();
            }

            ~BeamSearchDecoder() = default;

            void prepare_child_restart_shell_metadata() {
                this->child_restart_adjacent_checks_by_bit.assign(
                        this->bit_count,
                        std::vector<uint8_t>(this->check_count, 0));
                this->child_restart_shell2_bits_by_bit.assign(
                        this->bit_count,
                        std::vector<uint8_t>(this->bit_count, 0));

                for (int bit = 0; bit < this->bit_count; bit++) {
                    for (auto &e: this->pcm.iterate_column(bit)) {
                        const int check = e.row_index;
                        this->child_restart_adjacent_checks_by_bit[bit][check] = 1;
                    }
                    for (int check = 0; check < this->check_count; check++) {
                        if (!this->child_restart_adjacent_checks_by_bit[bit][check]) continue;
                        for (auto &e: this->pcm.iterate_row(check)) {
                            const int other_bit = e.col_index;
                            if (other_bit == bit) continue;
                            this->child_restart_shell2_bits_by_bit[bit][other_bit] = 1;
                        }
                    }
                }
            }

            inline int child_restart_shell_for_edge(int clamp_bit, int bit, int check) const {
                if (clamp_bit < 0 || clamp_bit >= this->bit_count) {
                    return 3;
                }
                if (bit == clamp_bit) {
                    return 1;
                }
                if (check >= 0 &&
                    check < this->check_count &&
                    this->child_restart_adjacent_checks_by_bit[clamp_bit][check]) {
                    return 1;
                }
                if (this->child_restart_shell2_bits_by_bit[clamp_bit][bit]) {
                    return 2;
                }
                return 3;
            }

            inline double child_restart_near_clamp_penalty(int flip_count, bool force_disagrees) const {
                if (!this->child_restart_adaptive_near_clamp) {
                    return 0.0;
                }
                double penalty = 0.0;
                if (this->child_restart_adaptive_flip_penalty > 0.0 && this->child_restart_adaptive_flip_threshold > 0) {
                    const double flip_frac = std::min(
                            1.0,
                            static_cast<double>(std::max(0, flip_count)) /
                                    static_cast<double>(this->child_restart_adaptive_flip_threshold));
                    penalty += this->child_restart_adaptive_flip_penalty * flip_frac;
                }
                if (force_disagrees) {
                    penalty += this->child_restart_adaptive_disagree_penalty;
                }
                return std::min(1.0, std::max(0.0, penalty));
            }

            inline double child_restart_alpha_for_edge(
                    int clamp_bit,
                    int bit,
                    int check,
                    int parent_flip_count,
                    bool force_disagrees) const {
                if (!this->child_restart_local_shells) {
                    return this->child_restart_alpha;
                }
                const int shell = this->child_restart_shell_for_edge(clamp_bit, bit, check);
                double alpha = this->child_restart_local_shell_alpha_far;
                if (shell == 1) {
                    alpha = this->child_restart_local_shell_alpha_radius1;
                } else if (shell == 2) {
                    alpha = this->child_restart_local_shell_alpha_radius2;
                }
                if (shell <= 2) {
                    alpha = std::max(0.0, alpha - this->child_restart_near_clamp_penalty(parent_flip_count, force_disagrees));
                }
                return alpha;
            }

            void initialise_log_domain_bp() {
                // initialise BP
                const uint64_t shot = this->decode_calls;
                for (int i = 0; i < this->bit_count; i++) {
                    double base_llr = std::log((1 - this->channel_probabilities[i]) / this->channel_probabilities[i]);
                    // Deterministic micro-jitter to break exact LLR ties on highly symmetric instances.
                    uint64_t h = (static_cast<uint64_t>(i) + 1ULL) * 0x9E3779B97F4A7C15ULL;
                    h ^= (shot + 1ULL) * 0xBF58476D1CE4E5B9ULL;
                    h ^= (h >> 30);
                    h *= 0xBF58476D1CE4E5B9ULL;
                    h ^= (h >> 27);
                    h *= 0x94D049BB133111EBULL;
                    h ^= (h >> 31);
                    double centered = (static_cast<double>(h & 0xFFFFULL) - 32768.0) / 32768.0;
                    this->initial_log_prob_ratios[i] = base_llr + (1e-10 * centered);

                    for (auto &e: this->pcm.iterate_column(i)) {
                        e.bit_to_check_msg = this->initial_log_prob_ratios[i];
                    }
                }
            }

            std::vector<uint8_t> &decode(std::vector<uint8_t> &syndrome) {

                this->converge = 0;
                this->decode_calls += 1;
                std::fill(this->decoding.begin(), this->decoding.end(), 0);

                this->initialise_log_domain_bp();

                using Pair = std::pair<double, int>;
                std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> min_pq;
                std::vector<int> bit_masks(this->bit_count, -1);
                std::vector<double> LLR_sums(this->bit_count, 0);
                int edge_msgs_length = 0;
                for (int i = 0; i < this->bit_count; i++) {
                    edge_msgs_length += this->pcm.iterate_column(i).entry_count;
                }
                std::vector<std::vector<double>> edge_msgs(4 * this->beam_width, std::vector<double>(edge_msgs_length, 0));
                std::vector<std::vector<int>> fixed_indices(4 * this->beam_width, std::vector<int>(this->max_rounds + 1, 0));
                std::vector<std::vector<int>> fixed_values(4 * this->beam_width, std::vector<int>(this->max_rounds + 1, 0));
                std::vector<int> converged_value(4 * this->beam_width, -1);
                std::vector<int> branch_flip_counts(4 * this->beam_width, 0);
                std::vector<int> branch_posterior_bits(4 * this->beam_width, 0);
                std::vector<int> explore_list;
                std::vector<uint8_t> cur_decoding(this->bit_count, 0);
                std::vector<int> sign_flip_counts(this->bit_count, 0);
                std::vector<int> prev_sign(this->bit_count, 0);
                std::vector<uint8_t> prev_sign_valid(this->bit_count, 0);
                int start = 0, next_start = 2 * this->beam_width;
                double min_dec_weight = std::numeric_limits<double>::max();
                double cur_dec_weight;
                int sol_so_far = 0;

                // initial bp iterations
                for (int it = 1; it <= this->initial_iters; it++) {
                    //check to bit updates
                    for (int i = 0; i < check_count; i++) {

                        this->candidate_syndrome[i] = 0;
                        int total_sgn = 0;
                        int sgn = 0;
                        total_sgn = syndrome[i];
                        double temp = std::numeric_limits<double>::max();

                        for (auto &e: this->pcm.iterate_row(i)) {
                            if (e.bit_to_check_msg < 0) {
                                total_sgn += 1;
                            }
                            e.check_to_bit_msg = temp;
                            double abs_bit_to_check_msg = std::abs(e.bit_to_check_msg);
                            if (abs_bit_to_check_msg < temp) {
                                temp = abs_bit_to_check_msg;
                            }
                        }

                        temp = std::numeric_limits<double>::max();
                        for (auto &e: this->pcm.reverse_iterate_row(i)) {
                            sgn = total_sgn;
                            if (e.bit_to_check_msg < 0) {
                                sgn += 1;
                            }
                            if (temp < e.check_to_bit_msg) {
                                e.check_to_bit_msg = temp;
                            }

                            int message_sign = (sgn % 2 == 0) ? 1.0 : -1.0;

                            e.check_to_bit_msg *= message_sign;

                            double abs_bit_to_check_msg = std::abs(e.bit_to_check_msg);
                            if (abs_bit_to_check_msg < temp) {
                                temp = abs_bit_to_check_msg;
                            }
                        }
                    }

                    //compute log probability ratios
                    for (int i = 0; i < this->bit_count; i++) {
                        double temp = initial_log_prob_ratios[i];
                        for (auto &e: this->pcm.iterate_column(i)) {
                            e.bit_to_check_msg = temp;
                            temp += e.check_to_bit_msg;
                        }

                        //make hard decision on basis of log probability ratio for bit i
                        this->log_prob_ratios[i] = temp;
                        const int cur_sign = (temp < 0) ? 1 : 0;
                        if (prev_sign_valid[i] && cur_sign != prev_sign[i]) {
                            sign_flip_counts[i] += 1;
                        }
                        prev_sign_valid[i] = 1;
                        prev_sign[i] = cur_sign;
                        if (temp < 0) {
                            this->decoding[i] = 1;
                            for (auto &e: this->pcm.iterate_column(i)) {
                                this->candidate_syndrome[e.row_index] ^= 1;
                            }
                        } else {
                            this->decoding[i] = 0;
                        }
                    }

                    //compute bit to check update
                    for (int i = 0; i < bit_count; i++) {
                        double temp = 0;
                        for (auto &e: this->pcm.reverse_iterate_column(i)) {
                            e.bit_to_check_msg += temp;
                            temp += e.check_to_bit_msg;
                        }
                    }

                    //calculate statistics
                    for (int i = 0; i < this->bit_count; i++) {
                        LLR_sums[i] += this->log_prob_ratios[i];
                    }

                    if (std::equal(candidate_syndrome.begin(), candidate_syndrome.end(), syndrome.begin())) {
                        this->converge = true;
                    }

                    this->iterations = it;

                    if (this->converge) {
                        min_dec_weight = 0;
                        for (int i = 0; i < this->bit_count; i++) {
                            if (this->decoding[i]) {
                                min_dec_weight += this->initial_log_prob_ratios[i];
                            }
                        }
                        sol_so_far++;
                        if (sol_so_far == this->num_results) {
                            return this->decoding;
                        }
                        break;
                    }
                }

                //Initialize for subsequent rounds
                double min_score = std::abs(LLR_sums[0]);
                int min_idx = 0;
                for (int i = 0; i < this->bit_count; i++) {
                    // skip qubits whose Tanner graph degree is <=2
                    if (this->pcm.iterate_column(i).entry_count <= 2) continue;
                    if (min_score > std::abs(LLR_sums[i])) {
                        min_score = std::abs(LLR_sums[i]);
                        min_idx = i;
                    }
                }
                fixed_indices[0][0] = min_idx;
                branch_flip_counts[0] = sign_flip_counts[min_idx];
                branch_posterior_bits[0] = this->decoding[min_idx];
                if (this->converge) {
                    converged_value[0] = cur_decoding[min_idx];
                } else {
                    converged_value[0] = -1;
                }
                int msg_idx = 0;
                for (int i = 0; i < this->bit_count; i++) {
                    for (auto &e: this->pcm.iterate_column(i)) {
                        edge_msgs[0][msg_idx] = e.bit_to_check_msg;
                        msg_idx++;
                    }
                }
                explore_list.push_back(0);

                // subsequent rounds
                for (int round = 0; round < this->max_rounds; round++) {
                    int store_idx = 0;

                    for (int path_id = 0; path_id < 2 * explore_list.size(); path_id++) {
                        int list_ele = explore_list[path_id / 2];
                        int bit_val = path_id % 2;
                        if (converged_value[start + list_ele] == bit_val) continue;
                        fixed_values[start + list_ele][round] = bit_val;
                        for (int i = 0; i <= round; i++) {
                            bit_masks[fixed_indices[start + list_ele][i]] = fixed_values[start + list_ele][i];
                            if (fixed_values[start + list_ele][i]) {
                                for (auto &e: this->pcm.iterate_column(fixed_indices[start + list_ele][i])) {
                                    syndrome[e.row_index] ^= 1;
                                }
                            }
                        }
                        // Warm-start from the parent path or cold-start from the channel prior.
                        const int clamp_bit = fixed_indices[start + list_ele][round];
                        const int parent_flip_count = branch_flip_counts[start + list_ele];
                        const bool force_disagrees = (bit_val != branch_posterior_bits[start + list_ele]);
                        sign_flip_counts.assign(this->bit_count, 0);
                        prev_sign.assign(this->bit_count, 0);
                        prev_sign_valid.assign(this->bit_count, 0);
                        msg_idx = 0;
                        for (int i = 0; i < this->bit_count; i++) {
                            if (bit_masks[i] != -1) {
                                msg_idx += this->pcm.iterate_column(i).entry_count;
                                continue;
                            }
                            for (auto &e: this->pcm.iterate_column(i)) {
                                const double alpha = this->child_restart_alpha_for_edge(
                                        clamp_bit,
                                        i,
                                        e.row_index,
                                        parent_flip_count,
                                        force_disagrees);
                                const double parent_msg = edge_msgs[start + list_ele][msg_idx];
                                const double base_msg = this->initial_log_prob_ratios[i];
                                e.bit_to_check_msg = (alpha * parent_msg) + ((1.0 - alpha) * base_msg);
                                msg_idx++;
                            }
                        }

                        this->converge = 0;
                        for (int i = 0; i < this->bit_count; i++) LLR_sums[i] = 0;

                        //main interation loop
                        for (int it = 1; it <= this->iters_per_round; it++) {
                            //check to bit updates
                            for (int i = 0; i < check_count; i++) {

                                this->candidate_syndrome[i] = 0;
                                int total_sgn = 0;
                                int sgn = 0;
                                total_sgn = syndrome[i];
                                double temp = std::numeric_limits<double>::max();

                                for (auto &e: this->pcm.iterate_row(i)) {
                                    // ignore fixed bits
                                    if (bit_masks[e.col_index] != -1) continue;
                                    if (e.bit_to_check_msg < 0) {
                                        total_sgn += 1;
                                    }
                                    e.check_to_bit_msg = temp;
                                    double abs_bit_to_check_msg = std::abs(e.bit_to_check_msg);
                                    if (abs_bit_to_check_msg < temp) {
                                        temp = abs_bit_to_check_msg;
                                    }
                                }

                                temp = std::numeric_limits<double>::max();
                                for (auto &e: this->pcm.reverse_iterate_row(i)) {
                                    // ignore fixed bits
                                    if (bit_masks[e.col_index] != -1) continue;
                                    sgn = total_sgn;
                                    if (e.bit_to_check_msg < 0) {
                                        sgn += 1;
                                    }
                                    if (temp < e.check_to_bit_msg) {
                                        e.check_to_bit_msg = temp;
                                    }

                                    int message_sign = (sgn % 2 == 0) ? 1.0 : -1.0;

                                    e.check_to_bit_msg *= message_sign;

                                    double abs_bit_to_check_msg = std::abs(e.bit_to_check_msg);
                                    if (abs_bit_to_check_msg < temp) {
                                        temp = abs_bit_to_check_msg;
                                    }
                                }
                            }

                            //compute log probability ratios
                            for (int i = 0; i < this->bit_count; i++) {
                                if (bit_masks[i] != -1) continue;
                                double temp = initial_log_prob_ratios[i];
                                for (auto &e: this->pcm.iterate_column(i)) {
                                    e.bit_to_check_msg = temp;
                                    temp += e.check_to_bit_msg;
                                }

                                //make hard decision on basis of log probability ratio for bit i
                                this->log_prob_ratios[i] = temp;
                                const int cur_sign = (temp < 0) ? 1 : 0;
                                if (prev_sign_valid[i] && cur_sign != prev_sign[i]) {
                                    sign_flip_counts[i] += 1;
                                }
                                prev_sign_valid[i] = 1;
                                prev_sign[i] = cur_sign;
                                if (temp < 0) {
                                    cur_decoding[i] = 1;
                                    for (auto &e: this->pcm.iterate_column(i)) {
                                        this->candidate_syndrome[e.row_index] ^= 1;
                                    }
                                } else {
                                    cur_decoding[i] = 0;
                                }
                            }

                            //compute bit to check update
                            for (int i = 0; i < bit_count; i++) {
                                if (bit_masks[i] != -1) continue;
                                double temp = 0;
                                for (auto &e: this->pcm.reverse_iterate_column(i)) {
                                    e.bit_to_check_msg += temp;
                                    temp += e.check_to_bit_msg;
                                }
                            }

                            //calculate statistics
                            for (int i = 0; i < this->bit_count; i++) {
                                if (bit_masks[i] == -1) {
                                    LLR_sums[i] += this->log_prob_ratios[i];
                                }
                            }

                            if (std::equal(candidate_syndrome.begin(), candidate_syndrome.end(), syndrome.begin())) {
                                this->converge = true;
                            }

                            this->iterations = it;

                            if (this->converge) {
                                cur_dec_weight = 0;
                                for (int i = 0; i < this->bit_count; i++) {
                                    if (bit_masks[i] != -1) {
                                        cur_decoding[i] = bit_masks[i];
                                    }
                                    if (cur_decoding[i]) {
                                        cur_dec_weight += this->initial_log_prob_ratios[i];
                                    }
                                }
                                if (cur_dec_weight < min_dec_weight) {
                                    min_dec_weight = cur_dec_weight;
                                    this->decoding = cur_decoding;
                                }
                                sol_so_far++;
                                if (sol_so_far == this->num_results) {
                                    return this->decoding;
                                }
                                break;
                            }
                        }

                        // Compare different paths.
                        double score = 0;
                        for (int i = 0; i < this->bit_count; i++) {
                            if (bit_masks[i] == -1) {
                                score += std::abs(LLR_sums[i]);
                            }
                        }
                        score = score / static_cast<double>(this->iterations);
                        if (min_pq.size() >= this->beam_width && score < min_pq.top().first) {
                            // restore the bit_masks vector and the syndrome vector
                            for (int i = 0; i <= round; i++) {
                                bit_masks[fixed_indices[start + list_ele][i]] = -1;
                                if (fixed_values[start + list_ele][i]) {
                                    for (auto &e: this->pcm.iterate_column(fixed_indices[start + list_ele][i])) {
                                        syndrome[e.row_index] ^= 1;
                                    }
                                }
                            }
                            continue;
                        }
                        for (int i = 0; i <= round; i++) {
                            fixed_indices[next_start + store_idx][i] = fixed_indices[start + list_ele][i];
                            fixed_values[next_start + store_idx][i] = fixed_values[start + list_ele][i];
                        }
                        min_score = std::numeric_limits<double>::max();
                        for (int i = 0; i < this->bit_count; i++) {
                            if (bit_masks[i] != -1) continue;
                            // skip qubits whose Tanner graph degree is <=2
                            if (this->pcm.iterate_column(i).entry_count <= 2) continue;
                            if (min_score > std::abs(LLR_sums[i])) {
                                min_score = std::abs(LLR_sums[i]);
                                min_idx = i;
                            }
                        }
                        fixed_indices[next_start + store_idx][round + 1] = min_idx;
                        branch_flip_counts[next_start + store_idx] = sign_flip_counts[min_idx];
                        branch_posterior_bits[next_start + store_idx] = cur_decoding[min_idx];
                        if (this->converge) {
                            converged_value[next_start + store_idx] = cur_decoding[min_idx];
                        } else {
                            converged_value[next_start + store_idx] = -1;
                        }

                        msg_idx = 0;
                        for (int i = 0; i < this->bit_count; i++) {
                            if (bit_masks[i] != -1) {
                                msg_idx += this->pcm.iterate_column(i).entry_count;
                                continue;
                            }
                            for (auto &e: this->pcm.iterate_column(i)) {
                                edge_msgs[next_start + store_idx][msg_idx] = e.bit_to_check_msg;
                                msg_idx++;
                            }
                        }

                        min_pq.push({score, store_idx});
                        if (min_pq.size() > this->beam_width) min_pq.pop();
                        store_idx++;

                        // restore the bit_masks vector and the syndrome vector
                        for (int i = 0; i <= round; i++) {
                            bit_masks[fixed_indices[start + list_ele][i]] = -1;
                            if (fixed_values[start + list_ele][i]) {
                                for (auto &e: this->pcm.iterate_column(fixed_indices[start + list_ele][i])) {
                                    syndrome[e.row_index] ^= 1;
                                }
                            }
                        }
                    }

                    // Update for next rounds. Take reverse order because we want to explore elements with higher scores first.
                    explore_list.resize(min_pq.size());
                    for (int i = explore_list.size() - 1; i >= 0; i--) {
                        explore_list[i] = min_pq.top().second;
                        min_pq.pop();
                    }

                    if (start == 0) {
                        start = 2 * this->beam_width;
                        next_start = 0;
                    } else {
                        start = 0;
                        next_start = 2 * this->beam_width;
                    }
                }
                return this->decoding;
            }
        };
    }
}  // namespace ldpc::bp

#endif
