import math

def gamma(z):
    """
    Wrapper around math.gamma (Python 3.8+) so that we can refer to it simply as gamma().
    """
    return math.gamma(z)

def t_pdf(u, m):
    """
    PDF of the Student t-distribution (not the CDF).
    t_pdf(u; m) = K_m * (1 + u^2 / m)^(-(m+1)/2)
    where
       K_m = Gamma((m+1)/2) / [ sqrt(m*pi) * Gamma(m/2) ].
    """
    # Compute K_m once
    K_m = gamma((m+1)/2.0) / ( math.sqrt(m*math.pi) * gamma(m/2.0) )
    return K_m * (1.0 + (u**2)/m)** (-(m+1)/2.0)

def simpson_integration(f, a, b, N=200):
    """
    Numerically integrate f(x) from x=a to x=b using Simpson's 1/3 rule with N subintervals.
    N must be even. If N is odd, we add 1.
    """
    if N % 2 != 0:
        N += 1

    h = (b - a) / N
    total = f(a) + f(b)

    # Sum terms
    odd_sum = 0.0
    even_sum = 0.0
    for i in range(1, N):
        x_i = a + i*h
        fx_i = f(x_i)
        if i % 2 == 1:
            odd_sum += fx_i
        else:
            even_sum += fx_i

    total += 4.0*odd_sum + 2.0*even_sum
    return (h/3.0)*total

def t_cdf(z, m):
    """
    Numerically approximate the CDF of the t-distribution with m degrees of freedom:
       F(z) = integral_{-infinity}^{z} t_pdf(u, m) du.
    We'll truncate "infinity" at a large negative number for practical integration.
    """
    # If z is very large negative, F(z) ~ 0. If z is very large positive, F(z) ~ 1.
    # We'll pick a sufficiently large negative bound, say -10 or -12 or even -20,
    # depending on how far out your table typically goes.
    NEG_BOUND = -10.0
    if z < -10:
        NEG_BOUND = min(z - 1.0, -10.0)  # if user picks z < -10, shift the bound a bit further

    # If z is very large, the integral from -∞ to z ~ 1, but we'll just do up to z.
    # If z < NEG_BOUND, we'll get a very small probability.
    def f(x):
        return t_pdf(x, m)

    return simpson_integration(f, NEG_BOUND, z, N=300)

def main():
    """
    Prompts user for m (degrees of freedom) and z, computes F(z).
    You can run multiple times and compare with Table A9.
    """
    print("This program computes the t-distribution CDF:")
    print("    F(z) = Km * ∫ from -∞ to z of (1 + u^2/m)^(-(m+1)/2) du\n")

    Again = True
    while Again:
        # 1) Ask for degrees of freedom, m
        m_str = input("Enter degrees of freedom (m): ").strip()
        if not m_str:
            print("No input received. Exiting.")
            break
        m = float(m_str)

        # 2) Ask for z
        z_str = input("Enter z-value: ").strip()
        if not z_str:
            print("No z given. Exiting.")
            break
        z_val = float(z_str)

        # 3) Compute CDF
        cdf_val = t_cdf(z_val, m)

        # 4) Display result
        print(f"\nFor m = {m}, z = {z_val},  F(z) ≈ {cdf_val:.6f}\n")

        # 5) Go again?
        again_str = input("Compute another? (y/n): ").lower().strip()
        if again_str not in ['y', 'yes']:
            Again = False

if __name__ == "__main__":
    main()
