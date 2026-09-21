#include <memory>
#include <stdexcept>

#include "afmexchange.hpp"
#include "antiferromagnet.hpp"
#include "dmi.hpp"
#include "energy.hpp"
#include "fieldquantity.hpp"
#include "fullmag.hpp"
#include "magnet.hpp"
#include "mumaxworld.hpp"
#include "neel.hpp"
#include "parameter.hpp"
#include "world.hpp"
#include "wrappers.hpp"

void wrap_antiferromagnet(py::module& m) {
  py::class_<Antiferromagnet, Magnet>(m, "Antiferromagnet")
      .def("sub1", &Antiferromagnet::sub1, py::return_value_policy::reference)
      .def("sub2", &Antiferromagnet::sub2, py::return_value_policy::reference)
      .def("sublattices", &Antiferromagnet::sublattices,
           py::return_value_policy::reference)
      .def(
          "other_sublattice",
          [](const Antiferromagnet* m, Ferromagnet* mag) {
            return m->getOtherSublattices(mag)[0];
          },
          py::return_value_policy::reference)
      .def_readonly("afmex_cell", &Antiferromagnet::afmex_cell)
      .def_readonly("afmex_nn", &Antiferromagnet::afmex_nn)
      .def_readonly("afmex_nn_dir", &Antiferromagnet::afmexNNDir)
      .def_readonly("inter_afmex_nn", &Antiferromagnet::interAfmExchNN)
      .def_readonly("scale_afmex_nn", &Antiferromagnet::scaleAfmExchNN)
      .def_readonly("latcon", &Antiferromagnet::latcon)
      .def_readonly("dmi_tensor", &Antiferromagnet::dmiTensor)
      .def_readonly("dmi_vector", &Antiferromagnet::dmiVector)

      .def("minimize", &Antiferromagnet::minimize, py::arg("tol"),
           py::arg("nsamples"))
      .def("relax", &Antiferromagnet::relax, py::arg("tol"))
      .def_readwrite("enable_cavity_afm", &Antiferromagnet::enableCavityAfm)
      .def_readwrite("cavity_omega", &Antiferromagnet::cavityOmega)
      .def_readwrite("cavity_kappa", &Antiferromagnet::cavityKappa)
      .def_readwrite("cavity_h0", &Antiferromagnet::cavityH0)
      .def_readwrite("cavity_energy_field", &Antiferromagnet::cavityEnergyField)
      .def_readwrite("cavity_drive_re", &Antiferromagnet::cavityDriveRe)
      .def_readwrite("cavity_drive_im", &Antiferromagnet::cavityDriveIm)
      .def_readwrite("enable_aux_mode", &Antiferromagnet::enableAuxMode)
      .def_readwrite("aux_omega", &Antiferromagnet::auxOmega)
      .def_readwrite("aux_kappa", &Antiferromagnet::auxKappa)
      .def_readwrite("aux_j", &Antiferromagnet::auxJ)
      .def_readwrite("aux_h0", &Antiferromagnet::auxH0)
      .def("cavity_amplitude", &Antiferromagnet::cavityAmplitude,
           py::return_value_policy::reference_internal)
      .def("aux_amplitude", &Antiferromagnet::auxAmplitude,
           py::return_value_policy::reference_internal);

  m.def("neel_vector", &neelVectorQuantity);
  m.def("full_magnetization",
        py::overload_cast<const Antiferromagnet*>(&fullMagnetizationQuantity));

  m.def("angle_field", &angleFieldQuantity);
  m.def("max_intracell_angle", &maxAngle);

  m.def("total_energy_density",
        py::overload_cast<const Antiferromagnet*>(&totalEnergyDensityQuantity));
  m.def("total_energy",
        py::overload_cast<const Antiferromagnet*>(&totalEnergyQuantity));
}
