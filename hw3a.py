# region imports
from numericalMethods import GPDF, Probability


# endregion

def main():
    """
    Modified program that:
     - Lets you pick 'p' to calculate Probability from a given c, OR
     - Lets you pick 'c' to calculate c from a given Probability.

    Retains your original one-sided vs. two-sided logic for how the Probability is calculated/printed.
    """
    # --- Default values ---
    mean = 0.0
    stDev = 1.0
    c_val = 0.5  # only relevant if user is finding p
    p_val = 0.45  # only relevant if user is finding c
    GT = False
    OneSided = True
    yesOptions = ["y", "yes", "true"]

    Again = True
    while Again:
        print("\nDo you want to find 'p' (probability) given c, OR find 'c' given p?")
        mode = input("Type 'p' or 'c': ").strip().lower()

        # 1) Ask for mean/stDev
        resp = input(f"Population mean? (default={mean:.3f}): ").strip()
        if resp:
            mean = float(resp)
        resp = input(f"Standard deviation? (default={stDev:.3f}): ").strip()
        if resp:
            stDev = float(resp)

        # 2) Ask if Probability is greater than c
        resp = input(f"Probability greater than c? (y/n) (default={GT}): ").strip().lower()
        if resp in yesOptions:
            GT = True
        elif resp != "":
            GT = False

        # 3) Ask if one-sided
        resp = input(f"One-sided? (y/n) (default={OneSided}): ").strip().lower()
        if resp in yesOptions:
            OneSided = True
        elif resp != "":
            OneSided = False

        if mode == 'p':
            ############################################################
            # USER WANTS TO FIND THE PROBABILITY GIVEN c
            ############################################################
            resp = input(f"Enter c (default={c_val:.3f}): ").strip()
            if resp:
                c_val = float(resp)

            # Now compute probability from c_val
            if OneSided:
                # Use Probability function directly
                result_p = Probability(GPDF, (mean, stDev), c_val, GT=GT)
                print(
                    f"\nResult => P(x{'>' if GT else '<'}{c_val:.2f} | mean={mean:.2f}, std={stDev:.2f}) = {result_p:.4f}")
            else:
                # TWO-SIDED logic: "prob = 1 - 2 * Probability(..., GT=True)" from your code
                p_inside = 1 - 2 * Probability(GPDF, (mean, stDev), c_val, GT=True)
                if GT:
                    # Original code prints the "outside" portion
                    print(
                        f"\nResult => P({mean - (c_val - mean):.2f} > x > {mean + (c_val - mean):.2f} | {mean:.2f}, {stDev:.2f}) = {1 - p_inside:.4f}")
                else:
                    print(
                        f"\nResult => P({mean - (c_val - mean):.2f} < x < {mean + (c_val - mean):.2f} | {mean:.2f}, {stDev:.2f}) = {p_inside:.4f}")

        elif mode == 'c':
            ############################################################
            # USER WANTS TO FIND c GIVEN A PROBABILITY
            ############################################################
            resp = input(f"Enter desired probability p (default={p_val:.3f}): ").strip()
            if resp:
                p_val = float(resp)

            # We'll do a simple bisection to find c in [mean-5*stDev, mean+5*stDev]
            left, right = mean - 5.0 * stDev, mean + 5.0 * stDev

            def f(c_test):
                if OneSided:
                    # Probability(...) with GT
                    p_test = Probability(GPDF, (mean, stDev), c_test, GT=GT)
                else:
                    # two-sided inside
                    p_inside = 1 - 2 * Probability(GPDF, (mean, stDev), c_test, GT=True)
                    p_test = (1 - p_inside) if GT else p_inside
                return p_test - p_val

            # Bisection
            for _ in range(50):
                mid = 0.5 * (left + right)
                if f(left) * f(mid) < 0:
                    right = mid
                else:
                    left = mid
                if abs(right - left) < 1e-7:
                    break
            c_solution = 0.5 * (left + right)

            # Display result
            # Compute final probability at that c for clarity
            final_p = f(c_solution) + p_val  # i.e. p_val + (p_test - p_val) = p_test
            if OneSided:
                print(f"\nFound c ≈ {c_solution:.4f} => Probability P(x{'>' if GT else '<'}c) ≈ {final_p:.4f}")
            else:
                if GT:
                    print(f"\nFound c ≈ {c_solution:.4f} => Outside Probability ≈ {final_p:.4f}")
                else:
                    print(f"\nFound c ≈ {c_solution:.4f} => Inside Probability ≈ {final_p:.4f}")

        else:
            print("Invalid choice (must be 'p' or 'c').")

        # Ask if user wants to go again
        resp = input("Go again? (y/n): ").strip().lower()
        Again = True if resp in yesOptions else False


if __name__ == "__main__":
    main()
