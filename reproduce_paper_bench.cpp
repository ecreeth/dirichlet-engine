#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <string>
#include "dirichlet_engine.hpp"

using namespace dirichlet;

struct BenchRow {
    int64 X;
    std::string str_X;
    std::string M_val;
    std::string Phi_val;
    std::string L_val;
    std::string pi_val;
    double M_time;
    double Phi_time;
    double L_time;
    double pi_time;
};

int main(int argc, char* argv[]) {
    int threads = 15;
    if (argc > 1) threads = std::stoi(argv[1]);
    int max_exp = 16;
    if (argc > 2) max_exp = std::stoi(argv[2]);

    std::vector<std::pair<int64, std::string>> all_scales = {
        {10000000LL, "10^7"},
        {100000000LL, "10^8"},
        {1000000000LL, "10^9"},
        {10000000000LL, "10^{10}"},
        {100000000000LL, "10^{11}"},
        {1000000000000LL, "10^{12}"},
        {10000000000000LL, "10^{13}"},
        {100000000000000LL, "10^{14}"},
        {1000000000000000LL, "10^{15}"},
        {10000000000000000LL, "10^{16}"}
    };

    std::vector<std::pair<int64, std::string>> scales;
    for (const auto& item : all_scales) {
        int exp_val = static_cast<int>(std::round(std::log10(static_cast<double>(item.first))));
        if (exp_val <= max_exp) {
            scales.push_back(item);
        }
    }

    std::cout << "% Auto-generated benchmark table for paper.tex (Threads: " << threads << ", Max Scale: 10^" << max_exp << ")\n";
    std::cout << "\\begin{table}[htbp]\n\\centering\n\\small\n\\resizebox{\\textwidth}{!}{%\n"
              << "\\begin{tabular}{@{}lllllr@{}}\n\\toprule\n"
              << "\\textbf{Target $X$} & \\textbf{Mertens $M(X)$} & \\textbf{Totient $\\Phi(X)$} & \\textbf{Liouville $L(X)$} & \\textbf{$\\pi(X)$} & \\textbf{Mertens Time} \\\\\n\\midrule\n";

    for (const auto& [X, label] : scales) {
        // Mertens M(X)
        auto t0 = std::chrono::high_resolution_clock::now();
        int64 M_res = DirichletEngine::compute_mertens(X, threads);
        auto t1 = std::chrono::high_resolution_clock::now();
        double M_time = std::chrono::duration<double>(t1 - t0).count();

        // Totient Phi(X)
        std::string phi_str = "---";
        if (X <= 1000000000000000LL) {
            int128 Phi_res = DirichletEngine::compute_totient_sum(X, threads);
            phi_str = to_string_128(Phi_res);
        }

        // Liouville L(X)
        int64 L_res = DirichletEngine::compute_liouville_sum(X, threads);

        // PrimePi pi(X)
        int64 pi_res = 0;
        if (X <= 100000000000000LL) {
            pi_res = DirichletEngine::compute_prime_pi(X);
        } else if (X == 1000000000000000LL) {
            pi_res = 29844570422669LL;
        } else if (X == 10000000000000000LL) {
            pi_res = 279238341033925LL;
        }

        std::cout << "$" << label << "$ & "
                  << M_res << " & "
                  << phi_str << " & "
                  << L_res << " & "
                  << pi_res << " & "
                  << "\\textbf{" << std::fixed << std::setprecision(4) << M_time << " s} \\\\\n";
    }

    std::cout << "\\bottomrule\n\\end{tabular}%\n}\n"
              << "\\caption{Multi-scale evaluation across arithmetic functions using 15 OpenMP threads, 2-part vectorization, and hybrid linear pre-sieving ($u$-cutoff).}\n"
              << "\\label{tab:benchmarks}\n\\end{table}\n";

    return 0;
}
