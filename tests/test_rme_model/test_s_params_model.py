import sys
from pathlib import Path

# add this folder to path to import the model file
LOCAL = Path(__file__).parent
TESTS_DIR = LOCAL.parent
OUTPUT = LOCAL / 'output'
sys.path.append(str(LOCAL))


# fails if on posix
def test_rmemodel():
    import pylab as pl  # noqa: E402
    import model  # noqa: E402
    import numpy as np  # noqa: E402
    from rmellipse.uobjects import Norm, Uniform, RMEModel  # noqa: E402

    # output data somewhere it won't get commited
    output_dir = OUTPUT
    output_dir.mkdir(exist_ok=True)
    RLGC_data = model.RLGC.load(LOCAL / 'data//Water_Q2D.csv')

    trials = 5
    line_length = 24140e-6  # in meters
    line_length_uncertainty = 2000e-6  # in meters
    length_parameter = Norm(
        'line length', loc=line_length, scale=line_length_uncertainty
    )

    nominal_sensitivity = 0.1
    sensitivity_uncertianty = 0.01
    sensitivity_parameter = Uniform(
        'Sensitivity', loc=nominal_sensitivity, scale=sensitivity_uncertianty
    )
    tunable_RLGC_model = RMEModel(model.tune_CG)
    voltage = np.linspace(0, 5, 6)
    tunable_RLGC_model.bind(
        RLGC_data=RLGC_data, voltage=voltage, sensitivity=sensitivity_parameter
    )
    tunable_RLGC_model.name = 'RLGC_model'

    # # Monte Carlo example

    S_params_model = RMEModel(model.get_S_parameters)
    S_params_model.name = 'S_params_model'

    S_params_model.setup(
        linear_uncertainty=False,
        offset=0,
        samples=trials,
    )

    S_params_MC_1 = S_params_model(
        RLGC_data=RLGC_data,
        length=length_parameter,
        Zr=50.0,
    )

    S_params_model.setup(linear_uncertainty=True)
    # required_coords={'frequency': S_params_MC.frequency})

    S_params_linear_1 = S_params_model(
        RLGC_data=RLGC_data,
        length=length_parameter,
        Zr=50.0,
    )

    S_params_model.setup(linear_uncertainty=True, required_coords={'voltage': voltage})
    # required_coords={'frequency': S_params_MC.frequency})

    S_params_linear_2 = S_params_model(
        RLGC_data=tunable_RLGC_model,
        length=length_parameter,
        Zr=50.0,
    )

    S_params_model.setup(
        linear_uncertainty=False,
        offset=0,
        samples=trials,
        required_coords={'voltage': voltage},
    )

    S_params_MC_2 = S_params_model(
        RLGC_data=RLGC_data,
        length=length_parameter,
        Zr=50.0,
    )

    # Plot MC analysis
    frequency = S_params_MC_1.frequency
    fig, ax = pl.subplots(2, figsize=(13.5, 6.6))
    # #ax.set_title("frequency domain")
    for i in range(0, trials):
        color = list(pl.rcParams['axes.prop_cycle'])[i]['color']
        S21 = S_params_MC_1.sel({'col': 'S21', 'sample_id': i})
        ax[0].plot(
            frequency / 1e9,
            20.0 * np.log10(np.abs(S21)),
            label='MC Trial {}'.format(i),
            color=color,
            linewidth=2,
        )
        ax[1].plot(
            frequency / 1e9,
            np.angle(S21, deg=True),
            label='MC Trial {}'.format(i),
            color=color,
            linewidth=2,
        )

    ax[0].legend()
    ax[0].set_ylabel(r'Mag($S_{21}$) (dB)')
    ax[1].set_ylabel(r'Phase($S_{21}$) (deg.)')
    ax[1].set_xlabel('Frequency (GHz)')
    pl.savefig(output_dir / 'S21_freq_MC.png')

    # plot linear analysis 1
    fig, ax = pl.subplots(2, figsize=(13.5, 6.6))
    umech_id = S_params_linear_1.coords['umech_id']
    for i, umech in enumerate(umech_id):
        color = list(pl.rcParams['axes.prop_cycle'])[i]['color']
        S21 = S_params_linear_1.sel({'col': 'S21', 'umech_id': umech})
        ax[0].plot(
            frequency / 1e9,
            20.0 * np.log10(np.abs(S21)),
            label=str(umech.data),
            color=color,
            linewidth=2,
        )
        ax[1].plot(
            frequency / 1e9,
            np.angle(S21, deg=True),
            label=str(umech.data),
            color=color,
            linewidth=2,
        )

    ax[0].legend()
    ax[0].set_ylabel(r'Mag($S_{21}$) (dB)')
    ax[1].set_ylabel(r'Phase($S_{21}$) (deg.)')
    ax[1].set_xlabel('Frequency (GHz)')
    pl.savefig(output_dir / 'S21_freq_linear.png')

    # plot linear analysis 2
    fig, ax = pl.subplots(2, figsize=(13.5, 6.6))
    for i, v in enumerate(voltage):
        S21 = S_params_linear_2.sel({'col': 'S21', 'umech_id': 'nominal', 'voltage': v})
        color = list(pl.rcParams['axes.prop_cycle'])[i]['color']
        ax[0].plot(
            frequency / 1e9,
            20.0 * np.log10(np.abs(S21)),
            label='voltage = ' + str(v),
            color=color,
            linewidth=2,
        )
        ax[1].plot(
            frequency / 1e9,
            np.angle(S21, deg=True),
            label='voltage = ' + str(v),
            color=color,
            linewidth=2,
        )

    ax[0].legend()
    ax[0].set_ylabel(r'Mag($S_{21}$) (dB)')
    ax[1].set_ylabel(r'Phase($S_{21}$) (deg.)')
    ax[1].set_xlabel('Frequency (GHz)')
    pl.savefig(output_dir / 'S21_freq_linear_voltage.png')
