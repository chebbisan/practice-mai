%% bench_matlab.m — Benchmark: full pipeline (AF + element pattern + normalization + directivity)
%% Usage: run in MATLAB/Octave

SPEED_OF_LIGHT = 3e8;
N_RUNS = 5;

%% ========== 1D: N=16, n_theta=1001, cosine element ==========
fprintf('============================================================\n');
fprintf('MATLAB/Octave benchmark\n');
fprintf('============================================================\n\n');

N = 16;
freq_hz = 3e9;
n_theta = 1001;

lambda_ = SPEED_OF_LIGHT / freq_hz;
d = lambda_ / 2;
k = 2 * pi / lambda_;
L = d * (N - 1);
x_arr = (0:N-1) * d - L / 2;
amplitudes = ones(1, N);
theta = linspace(-pi/2, pi/2, n_theta);

% Warmup
[D0_1d, ~] = pipeline_1d(N, x_arr, amplitudes, theta, k, n_theta);

times_1d = zeros(1, N_RUNS);
for r = 1:N_RUNS
    tic;
    [D0_1d, ~] = pipeline_1d(N, x_arr, amplitudes, theta, k, n_theta);
    times_1d(r) = toc;
end

fprintf('1D: N=%d, n_theta=%d, cosine element\n', N, n_theta);
fprintf('  avg=%.4f s  best=%.4f s  D0=%.2f (%.2f dB)\n\n', ...
    mean(times_1d), min(times_1d), D0_1d, 10*log10(D0_1d));

%% ========== 2D: 40x12, n_theta=101, n_phi=101 ==========
Nx = 40; Ny = 12;
n_theta_2d = 101; n_phi_2d = 101;

N2 = Nx * Ny;
x2 = zeros(1, N2);
y2 = zeros(1, N2);
idx = 1;
for ix = 0:Nx-1
    for iy = 0:Ny-1
        x2(idx) = (ix - (Nx-1)/2) * d;
        y2(idx) = (iy - (Ny-1)/2) * d;
        idx = idx + 1;
    end
end
amplitudes2 = ones(1, N2);
theta2 = linspace(-pi/2, pi/2, n_theta_2d);
phi2 = linspace(-pi/2, pi/2, n_phi_2d);

% Warmup
[D0_2d, ~] = pipeline_2d(N2, x2, y2, amplitudes2, theta2, phi2, k, n_theta_2d, n_phi_2d);

times_2d = zeros(1, N_RUNS);
for r = 1:N_RUNS
    tic;
    [D0_2d, ~] = pipeline_2d(N2, x2, y2, amplitudes2, theta2, phi2, k, n_theta_2d, n_phi_2d);
    times_2d(r) = toc;
end

fprintf('2D: %dx%d, n_theta=%d, n_phi=%d, cosine element\n', Nx, Ny, n_theta_2d, n_phi_2d);
fprintf('  avg=%.4f s  best=%.4f s  D0=%.2f (%.2f dB)\n\n', ...
    mean(times_2d), min(times_2d), D0_2d, 10*log10(D0_2d));

fprintf('Summary:\n');
fprintf('  1D: %.4f s\n', mean(times_1d));
fprintf('  2D: %.4f s\n', mean(times_2d));


%% ==================== Functions ====================

function [D0, full] = pipeline_1d(N, x_arr, amplitudes, theta, k, n_theta)
    % Array factor — vectorized
    af = zeros(1, n_theta);
    for i = 1:n_theta
        sin_t = sin(theta(i));
        re = 0; im = 0;
        for j = 1:N
            phase = -k * x_arr(j) * sin_t;
            re = re + amplitudes(j) * cos(phase);
            im = im + amplitudes(j) * sin(phase);
        end
        af(i) = sqrt(re^2 + im^2) / N;
    end

    % Element pattern (cosine)
    f1 = abs(cos(theta));

    % Full pattern, normalize
    full = f1 .* af;
    full = full / max(full);

    % Directivity (trapezoidal)
    D0 = 2.0 / trapz(theta, full.^2 .* cos(theta));
end

function [D0, full] = pipeline_2d(N, x_arr, y_arr, amplitudes, theta, phi, k, n_theta, n_phi)
    % Array factor
    af = zeros(n_theta, n_phi);
    for i = 1:n_theta
        sin_t = sin(theta(i));
        for j = 1:n_phi
            cos_p = cos(phi(j));
            sin_p = sin(phi(j));
            re = 0; im = 0;
            for n = 1:N
                phase = -k * (x_arr(n) * sin_t * cos_p + y_arr(n) * sin_t * sin_p);
                re = re + amplitudes(n) * cos(phase);
                im = im + amplitudes(n) * sin(phase);
            end
            af(i, j) = sqrt(re^2 + im^2) / N;
        end
    end

    % Element pattern (cosine), broadcast
    f1 = abs(cos(theta'));  % column vector
    full = f1 .* af;
    full = full / max(full(:));

    % Directivity 2D
    integrand = full.^2 .* cos(theta');
    inner = trapz(theta, integrand, 1);  % integrate along theta for each phi
    D0 = 4 * pi / trapz(phi, inner);
end
