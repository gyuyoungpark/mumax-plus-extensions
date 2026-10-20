STORED RAW ANALYSIS INTERFACE -- PORTABLE SOURCE ONLY

This source-only port does not repair or revalidate historical public figures.
Its local checks are standard-library metadata/I/O controls, not new FFTs,
fits, scientific predictions or simulations. The original frozen source had
finite mathematical/software controls and one lossless CPU FM pilot with two
references. Those observations concern the original exact hashes, not this
edited port. No native/GPU validation is claimed.

The raw complex carrier convention is exp(-i 2*pi*f*t). NumPy's FFT convention
therefore requires physical_frequency_hz = -numpy.fft.fftfreq(...). The complete
complex FFT is retained in its original order and in ascending physical-frequency
order. Neither half is discarded. A real rfft axis is never attached to a
truncated complex FFT.

Time checks require finite, strictly increasing timestamps and explicitly named
absolute/relative uniformity tolerances. Accepted clock deviations are recorded;
no interpolation, resampling or silent retry occurs. The original time array and
its storage dtype are preserved. Validation uses float64 arithmetic but does not
pretend a float32 source clock was recorded at float64 precision. The raw manifest
must declare sampling.time_dtype, sampling.signal_dtype and
sampling.timestamps_interpolated=false. Unsupported/missing provenance fails.

Raw arrays must include a complex time trajectory or both explicit real and
imaginary quadratures. Split quadratures require --imaginary-sign +1 or -1:
  +1: real + i*imaginary, for example q+i*p
  -1: real - i*imaginary, for example mx-i*my
The choice and source keys are retained. A complex-key input is already assembled
and must have its convention declared in the raw manifest. Keys named power,
spectrum, spectra, FFT or frequency are refused as raw signals. A missing phase,
discarded frequency half or missing time history cannot be reconstructed from
the retained public power maps by this interface. NumPy loads always use
allow_pickle=False. No object array is deserialized.

Window coefficients, coherent gain, ENBW, raw complex samples, windowed complex
samples, all complex FFT coefficients and normalization metadata are preserved.
Power is abs(FFT/(Nraw*coherent_gain))**2, without a one-sided factor of two.
The metadata distinguishes:
  - nominal independent bin scale: 1/(Nraw*dt), also called 1/T here;
  - FFT grid spacing: 1/(Nfft*dt);
  - registered conservative resolution criterion: max(1/T, window ENBW);
  - registered frequency uncertainty floor: half that criterion.
The last two are declared analysis rules, not universal resolution theorems or
fitted uncertainty estimates. Zero padding refines the sampling grid only.

Peak selection uses a physical band and minimum separation in Hz, never a fixed
distance of five bins. Its default separation also respects the registered window
criterion. A lower criterion can be explicitly requested for diagnostics, but the
result continues to report unresolved pairs. Peak existence alone does not prove
hybridization or a measured linewidth. No linewidth fitting implementation exists.

Expected branch frequencies must be supplied externally; the module generates no
theory/eigenvalues. A one-to-one minimum-cost assignment maximizes in-tolerance
coverage before minimizing residuals. One observed peak cannot satisfy two
branches. Every report includes missing branches, coverage, unused peaks, and a
conditional matched-only RMSE. Missing coverage, unresolved expected pairs, or an
assignment tolerance below the registered uncertainty floor cannot pass validation.

COMMANDS THAT ARE SAFE TO REVIEW

All --help paths parse arguments before importing the package or NumPy. They do
not read scientific inputs, perform FFTs, import a native runtime or execute cases.
Use the selected Python interpreter, from extensions/photon-magnon/methods:

  python analysis/analyze_raw.py --help
  python analysis/analyze_raw.py full-spectrum --help
  python analysis/analyze_raw.py source-identities

source-identities hashes analysis/analyze_raw.py and every .py source under
src/cavity_method, including __init__ and its local import dependencies. Any
added/changed package source invalidates the exact source map in an approval.

FUTURE RAW INSPECTION: NO FFT EXECUTION

The following command is documentation, not an executed action. It requires an
actual raw record and a matching original-run provenance manifest. It validates
time/complex storage only and reports fft_executed=false:

  python analysis/analyze_raw.py inspect-raw --input PATH_TO_RAW.npz --raw-manifest PATH_TO_RAW_MANIFEST.json --raw-label raw.npz --time-key time_s --real-key mx --imaginary-key my --imaginary-sign -1 --clock-rtol EXPLICIT_TOLERANCE --clock-atol-s EXPLICIT_TOLERANCE_SECONDS

The original-run manifest schema is version1, record_kind=raw_trajectory. It must
contain sources and binaries (portable labels mapped to sha256/bytes), raw_files
(including the exact supplied raw-label/hash/size), units (time=s and signal unit),
precision (float32/float64/complex64/complex128/mixed), parameters, sampling,
native_binary_use=declared, and time_convention="exp(-i 2*pi*f*t)".
It must also include signal_declarations. For example:
  "signal_declarations": {
    "m_minus": {
      "raw_label":"raw.npz", "time_key":"time_s",
      "real_key":"mx", "imaginary_key":"my", "imaginary_sign":-1,
      "carrier_convention":"exp(-i 2*pi*f*t)", "signal_unit":"1"
    }
  }
For an assembled complex trace, replace the split key/sign fields with
"complex_key":"a". The selected keys, sign, units and convention must match
exactly one original signal declaration. They cannot be chosen differently just
because a convenient alternative channel also exists in the same NPZ.
This schema cannot authenticate a historical source/binary pair that was never
recorded. Do not invent a manifest to make legacy files pass. A later imported
record may be documented separately, but it is not original-run provenance.

FUTURE APPROVED FOURIER ANALYSIS: EXPLICIT SEPARATE AUTHORIZATION

No command automatically runs a case or discovers a binary. Full analysis requires
both --execute-approved-analysis and an operator-supplied authorization JSON tied
to the exact raw record, raw manifest and current analysis source hashes. The
review record is not a cryptographic proof of human identity. It must refer to an
actual human approval. Stored-data analysis approval never grants permission for
new simulations. The authorization schema is:

  {
    "schema_version": 1,
    "authorization_kind": "stored_raw_analysis",
    "new_simulation_authorized": false,
    "approval_reference": "ACTUAL HUMAN APPROVAL REFERENCE",
    "operations": ["full_complex_fft"],
    "raw_input_sha256": "REPLACE WITH ACTUAL LOWERCASE SHA256",
    "raw_manifest_sha256": "REPLACE WITH ACTUAL LOWERCASE SHA256",
    "analysis_sources_sha256": {
      "analysis/analyze_raw.py": "ACTUAL SHA256",
      "src/cavity_method/__init__.py": "ACTUAL SHA256",
      "src/cavity_method/provenance.py": "ACTUAL SHA256",
      "src/cavity_method/spectrum.py": "ACTUAL SHA256"
    },
    "analysis_configuration_sha256": "ACTUAL CANONICAL CONFIGURATION SHA256",
    "additional_inputs_sha256": {}
  }

These placeholders are intentionally invalid. The software never writes its own
approval record. Any code/raw/manifest change invalidates the reviewed hashes.
The source map must contain ALL files printed by source-identities, including
energy/normalization modules if present. The schematic four-entry example above
is incomplete when additional local package dependencies exist and will fail.

To prepare a concrete review record without any Fourier execution, use the same
full-spectrum arguments with --print-review-configuration, omitting
--execute-approved-analysis and --authorization. This reads/hashes the named raw
file/manifest and source identities, prints the complete canonical configuration
and its digest, and reports approval_created=false and fft_executed=false. It
does not read scientific arrays, run NumPy, create approval or authorize itself.
The configuration binds raw keys, assembly sign, carrier, window, FFT length,
clock tolerances, peak band/separation/height, diagnostic resolution policy,
external branch input label, assignment tolerance and requested operations.
Changing any of these settings requires a new explicit review/approval digest.

After separately authorized inspection/review, and after explicitly creating a
new output directory under methods/analysis_outputs, an example command
for one approved record is:

  python analysis/analyze_raw.py full-spectrum --input PATH_TO_RAW.npz --raw-manifest PATH_TO_RAW_MANIFEST.json --raw-label raw.npz --time-key time_s --complex-key a --clock-rtol EXPLICIT_TOLERANCE --clock-atol-s EXPLICIT_TOLERANCE_SECONDS --window hann --authorization PATH_TO_APPROVAL.json --execute-approved-analysis --output analysis_outputs/APPROVED_NEW_RUN/full_spectrum.npz

For optional peaks, add --peak-band-hz MIN_HZ MAX_HZ, --minimum-separation-hz VALUE
and --minimum-relative-height VALUE, and explicitly authorize peak_selection.
For optional one-to-one coverage, also provide --expected-branches-json FILE and
--assignment-tolerance-hz VALUE; authorize branch_coverage and bind that file's
SHA256 under additional_inputs_sha256["expected_branches.json"]. Its content is:
  {"frequency_unit":"Hz", "branches":{"lower":EXTERNAL_VALUE,"upper":EXTERNAL_VALUE}}
No theoretical values are calculated by this tool.

Outputs are new files only, confined to analysis_outputs. Each file is published
atomically; overwrite is refused. A complete manifest is written last and hashes
the NPZ/analysis receipt. A partial multi-file failure leaves no complete receipt;
the existence of an orphan NPZ is not a completed/resumable analysis.
Before publishing that complete receipt, final raw/source identities must still
equal the held original raw manifest and approved source map. A mid-analysis
change refuses completion instead of labeling old computed arrays with new hashes.
Checkpoint utilities verify exact input-manifest and output hashes, but never
execute a resume action. Explicitly supplied native binaries can be hashed as
files; they are not imported, loaded or run. Manifests do not auto-export machine
paths, environment variables, credentials or secrets.
Checkpoint creation refuses an existing file by default. An explicit update is
allowed only for an existing checkpoint with matching schema, job and input
identity. It cannot silently replace an unrelated output or another job.
External expected-branch JSON is read once into held bytes, checked against the
approved digest, and then used without rereading it after the FFT.
Expected AND observed/matched peak separations must satisfy the registered window
criterion before coverage can pass, even when diagnostic narrow-peak selection
was explicitly allowed. Output precision is labeled mixed, with source precision
and actual FFT output dtype recorded separately.

The authorization enforcement boundary is the CLI. Low-level Python functions
are callable numerical/library utilities and do not independently prove human
approval. Their caller is responsible for separate authorization; importing them
or passing a dictionary must never be represented as an approval event.

VALIDATION STATUS

tests/test_analysis_metadata_io.py and tests/test_metadata_io_only.py contain
standard-library schema, hash, atomic I/O and authorization checks. They also
cover inert callback failure and AFM partner-metadata guards, without stepping
an engine or evaluating a physical model. No dynamics or FFT test sources are
distributed in this port. The original 31 mathematical/software controls
included 15 finite spectral controls; their acceptance concerns the original
frozen hashes and does not revalidate this edited port or historical figures.
No physical, native or GPU validation was executed for this port.

The old published spectral arrays and figures remain unrepaired pending recovery
of original raw trajectories/provenance or separately authorized new runs.
