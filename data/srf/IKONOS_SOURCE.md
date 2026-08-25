# IKONOS SRF provenance

`ikonos_relative_spectral_response.csv` is a text conversion of the numerical
`ikonos_sp` array distributed by the open-source HySure project:

- source repository: https://github.com/alfaiate/HySure
- source file: `data/ikonos_spec_resp.mat`
- source blob SHA: `6d60df334687adaaafd826cd215ecfb17f789bc4`
- array layout documented by HySure: wavelength, pan, blue, green, red, NIR
- wavelength sampling: 350-1035 nm in 5 nm increments

The same binary SRF blob (same SHA) is also used by the Hipandas repository,
which loads the variable `ikonos_sp` for IKONOS response simulation.

HySure's Pavia/ROSIS demo explicitly maps the IKONOS curves into the nominal
ROSIS 430-860 nm spectral support before resampling them to the HSI band count.
The repository follows that benchmark convention for the default PaviaU IKONOS
simulation and keeps the old `PaviaU.txt` 430-838 grid only for legacy checks.

The executable SRF pipeline still computes full-response energy coverage before
normalization. With the nominal 103-band 430-860 nm PaviaU grid and the 90%
threshold, IKONOS Blue, Green, Red and NIR are all retained. The NIR overlap is
about 92.75%; on the legacy 430-838 grid it is only about 81.22%, so that legacy
grid is not used for the default IKONOS simulation.
