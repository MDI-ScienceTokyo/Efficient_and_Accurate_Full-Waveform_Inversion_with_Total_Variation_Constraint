from pathlib import Path
import runpy


def main() -> None:
    marmousi_script = Path(__file__).with_name("box-TV-constrained-FWI-marmousi.py")
    marmousi = runpy.run_path(str(marmousi_script))

    marmousi["run_marmousi_experiments"](
        max_n_iters=10,
        n_shots=3,
        n_receivers=101,
        noise_sigma=0.0,
        gamma1=1e-6,
        gamma2=100,
        alpha_scales=(0.8,),
        include_gradient_baseline=False,
        result_root_path=Path("results/marmousi/noiseless"),
        random_seed=0,
    )


if __name__ == "__main__":
    main()
